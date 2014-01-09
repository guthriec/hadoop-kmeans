import org.apache.hadoop.io.*;

public class DoubleArrayWritable extends ArrayWritable {
	public DoubleArrayWritable() {
		super(DoubleWritable.class);
	}
	public DoubleArrayWritable(DoubleWritable[] values) {
		super(LongWritable.class, values);
	}
};
