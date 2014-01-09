import java.io.*;
import java.net.*;
import java.util.Iterator;
import java.lang.Math;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.filecache.*;

public class FindClusters {
	
	static int k = 10;
	static int numIterations = 20;
	static int dimension = 58;
	static double cost = 0;
	
    //euclidean distance calculation
	static double dist(double[] p1, double[] p2) {
		assert (p1.length == dimension);
		assert (p2.length == dimension);
		double sumSquares = 0;
		for (int i=0; i < dimension; i++) {
			double diff = p1[i]-p2[i];
			sumSquares += diff*diff;
		}
		
		return Math.sqrt(sumSquares);
	}
	
	
	public static class Map extends Mapper<LongWritable, Text, IntWritable, DoubleArrayWritable> {
        
		static double[][] centers = new double[k][dimension];
		
        //loads current centers
		public void setup(Context context) throws IOException {
			URI[] cacheFiles = DistributedCache.getCacheFiles(context.getConfiguration());
			FileSystem fs = FileSystem.get(context.getConfiguration());
			FSDataInputStream in = fs.open(new Path(cacheFiles[0]));
			BufferedReader br = new BufferedReader(new InputStreamReader(in));
			String currLine;
			for (int i=0; (currLine = br.readLine()) != null; i++) {
				String[] tokenized = currLine.split(new String(" "));
				assert (tokenized.length == dimension); 
				for (int j=0; j < dimension; j++)
					centers[i][j] = Double.parseDouble(tokenized[j]);
			}
		}
		
        /* 
         * strategy: calculate the minimum distance from each point in the value
         * string to some center. then write the index of the center that the point
         * now belongs to, along with the point itself. Also write each original
         * center with a flag to indicate that it's a center, not a data point
         * I messed up a little bit with writing the DoubleArrayWritables -
         * in retrospect I should've just written the index of the data point and
         * made the reducer look in the data file for the coordinates.
         */
		public void map(LongWritable key, Text value, Context context) 
				throws IOException, InterruptedException {
			double[] doc = new double[dimension];
			String[] docString = value.toString().split(new String(" "));
			assert(docString.length == dimension);
			
			for (int i=0; i < dimension; i++)
				doc[i] = Double.parseDouble(docString[i]);
			
			
			double[] distances = new double[k];
			for (int i=0; i < k; i++) 
				distances[i] = dist(doc, centers[i]);
			

			//find closest center (call it minIndex)
			int minIndex = -1;
			double minDistance = -1;
			for (int i=0; i < k; i++) {
				if (distances[i] > minDistance) {
					minIndex = i;
					minDistance = distances[i];
				}
			}
                    
            //this is how much adding this point costs
			cost += minDistance*minDistance;
			
            // we need to write each center. DoubleArrayWritable is the type we use
            // (see the other file, ArrayWritable man pages for how this works)
			DoubleWritable[] dummy = new DoubleWritable[dimension + 1];
			for (int i=0; i < k; i++) {
				for (int j=0; j < dimension; j++) 
					dummy[j] = new DoubleWritable(centers[i][j]);
			
				dummy[dimension] = new DoubleWritable(1);
				context.write(new IntWritable(i), new DoubleArrayWritable(dummy));
			}
			
            //write each (center, point) pair
			DoubleWritable[] docWritable = new DoubleWritable[dimension + 1]; 
			for (int i=0; i < dimension; i++) 
				docWritable[i] = new DoubleWritable(doc[i]);
			
			docWritable[dimension] = new DoubleWritable(0);
			
			context.write(new IntWritable(minIndex), new DoubleArrayWritable(docWritable));
		}
	
	} 
        
	public static class Reduce extends Reducer<IntWritable, DoubleArrayWritable, NullWritable, Text> {
		
		/*
         * for each center, we add up all the points represented by the DoubleArrayWritable, checking
         * to ensure that each point is actually a point, then we divide by the total number of 
         */
		public void reduce(IntWritable key, Iterable<DoubleArrayWritable> values, Context context) 
				throws IOException, InterruptedException {
			
			Iterator<DoubleArrayWritable> it = values.iterator();
			double[] pointSum = new double[dimension];
			for (int i=0; i < dimension; i++)
				pointSum[i] = 0;
			
			double[] point = new double[dimension+1];
			DoubleArrayWritable pointWritable = new DoubleArrayWritable();
			int total = 0;
			while (it.hasNext()) {
				pointWritable = it.next();
				for (int i=0; i <= dimension; i++)
					point[i] = ((DoubleWritable[])pointWritable.toArray())[i].get();
				
				double dummy = point[dimension];        //is this point actually a center?
				if (dummy < 0.5) {
					for (int i=0; i < dimension; i++) {
						pointSum[i] += point[i];
					}
                    total++;
				}
			}
			
			int hasAny = 0;
			for (int i=0; i < dimension; i++) {
				if (pointSum[i] > 0.00001) hasAny = 1;
			}
					
			double[] toOutput = new double[dimension];
			if (hasAny == 0) {                          //are there any real points in this cluster? if not, just leave it
				for (int i=0; i < dimension; i++) {
					toOutput[i] = point[i];
				}
			}
			else {                                      //here we compute the centroid
				for (int i=0; i < dimension; i++) {
					toOutput[i] = pointSum[i]/total;
				}
			}
			String outString = new String();
			for (int i=0; i < dimension; i++) {
				outString += String.format("%.3f",toOutput[i])+ " ";
			}
			context.write(NullWritable.get(), new Text(outString));
		}
	}

	public static void main(String[] args) throws Exception {
		double[] costArr = new double[numIterations+1];
		for (int i=0; i <= numIterations; i++) {
			cost = 0;
			String clusterFileName = new String("c1.txt");
			if (i > 0) clusterFileName = new String(i + "-roundtwo" + "/part-r-00000");
			Configuration conf = new Configuration();
			Job job = new Job(conf, "FindClusters");
			
            //all mappers need access to the set of clusters
			DistributedCache.addCacheFile(new Path("/home/cs246/kmeans-data/" + clusterFileName).toUri(), job.getConfiguration());
			
			job.setJarByClass(FindClusters.class);
			
			job.setMapOutputKeyClass(IntWritable.class);
			job.setMapOutputValueClass(DoubleArrayWritable.class);
			
			job.setOutputKeyClass(NullWritable.class);
			job.setOutputValueClass(Text.class);
		        
		    job.setMapperClass(Map.class);
		    job.setReducerClass(Reduce.class);
		        
		    job.setInputFormatClass(TextInputFormat.class);
		    job.setOutputFormatClass(TextOutputFormat.class);
		        
		    FileInputFormat.addInputPath(job, new Path(args[0]));
		    FileOutputFormat.setOutputPath(job, new Path("/home/cs246/kmeans-data/" + (i+1)+"-roundtwo"));
		        
		    job.waitForCompletion(true);
		    costArr[i] = cost;
		}
        
		for (int i=0; i <= numIterations; i++) System.out.println(costArr[i]);
	}
}
