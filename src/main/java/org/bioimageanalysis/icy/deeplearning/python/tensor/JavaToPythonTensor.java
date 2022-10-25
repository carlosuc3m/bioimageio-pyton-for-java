package org.bioimageanalysis.icy.deeplearning.python.tensor;

import java.util.stream.IntStream;

import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;
import org.bioimageanalysis.icy.deeplearning.utils.IndexingUtils;

import jep.NDArray;
import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;
import net.imglib2.view.IntervalView;

/**
 * Class that converts Java tensors into objects that can easily be translated into Python BioImage.io
 * tensors using JEP
 * @author Carlos Garcia Lopez de Haro
 *
 */

public class JavaToPythonTensor {
	
	/**
	 * Method that converts a Java tensor from {@link Tensor} into an object that can easily be converted into a Python
	 * tensor used by the BioImage.i core in Pyhton
	 * @param <T>
	 * @param javaTensor
	 * 	tensor object in Java
	 * @return tensor object ready to be converted to Python
	 */
	public static < T extends RealType< T > & NativeType< T > > PythonTensor fromJavaTensor( Tensor< T > javaTensor) {
		RandomAccessibleInterval<T> data = javaTensor.getData();
		T dt = Util.getTypeFromInterval(data);
		if (dt instanceof FloatType) {
			NDArray<float[]> nd = buildFromTensorFloat((RandomAccessibleInterval<FloatType>) javaTensor.getData());
			return new PythonTensor(javaTensor.getName(), javaTensor.getAxesOrderString(), nd);
		} else if (dt instanceof IntType) {
			NDArray<int[]> nd = buildFromTensorInt((RandomAccessibleInterval<IntType>) javaTensor.getData());
			return new PythonTensor(javaTensor.getName(), javaTensor.getAxesOrderString(), nd);
		} else if (dt instanceof DoubleType) {
			NDArray<double[]> nd = buildFromTensorDouble((RandomAccessibleInterval<DoubleType>) javaTensor.getData());
			return new PythonTensor(javaTensor.getName(), javaTensor.getAxesOrderString(), nd);
		} else if (dt instanceof LongType) {
			NDArray<long[]> nd = buildFromTensorLong((RandomAccessibleInterval<LongType>) javaTensor.getData());
			return new PythonTensor(javaTensor.getName(), javaTensor.getAxesOrderString(), nd);
			
		} else if (dt instanceof ByteType) {
			NDArray<byte[]> nd = buildFromTensorByte((RandomAccessibleInterval<ByteType>) javaTensor.getData());
			return new PythonTensor(javaTensor.getName(), javaTensor.getAxesOrderString(), nd);
		} else {
			throw new IllegalArgumentException("Conversion into Python of tensors with dsta type: '" + dt.getClass()+ "' "
					+ "is not supported.");
		}
	}

    /**
     * Builds a {@link NDArray} from a unsigned byte-typed {@link RandomAccessibleInterval}.
     * 
     * @param javaTensor
     *        The tensor data is read from.
     * @return The NDArray built from the tensor of type {@link DataType#UBYTE}.
     */
    private static NDArray<byte[]> buildFromTensorByte(RandomAccessibleInterval<ByteType> javaTensor)
    {
    	long[] tensorShape = javaTensor.dimensionsAsLongArray();
    	Cursor<ByteType> tensorCursor;
		if (javaTensor instanceof IntervalView)
			tensorCursor = ((IntervalView<ByteType>) javaTensor).cursor();
		else if (javaTensor instanceof Img)
			tensorCursor = ((Img<ByteType>) javaTensor).cursor();
		else
			throw new IllegalArgumentException("The data of the " + Tensor.class + " has "
					+ "to be an instance of " + Img.class + " or " + IntervalView.class);
		long flatSize = 1;
		for (long dd : javaTensor.dimensionsAsLongArray()) { flatSize *= dd;}
		byte[] flatArr = new byte[(int) flatSize];
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	byte val = tensorCursor.get().getByte();
        	flatArr[flatPos] = val;
		}
		int[] shape = IntStream.range(0, javaTensor.numDimensions()).map(i -> (int) javaTensor.dimension(i)).toArray();
		NDArray<byte[]> nd = new NDArray<byte[]>(flatArr, shape);
	 	return nd;
	}

    /**
     * Builds a {@link NDArray} from a unsigned integer-typed {@link RandomAccessibleInterval}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The NDArray built from the tensor of type {@link DataType#INT}.
     */
    private static NDArray<int[]> buildFromTensorInt(RandomAccessibleInterval<IntType> tensor)
    {
    	long[] tensorShape = tensor.dimensionsAsLongArray();
    	Cursor<IntType> tensorCursor;
		if (tensor instanceof IntervalView)
			tensorCursor = ((IntervalView<IntType>) tensor).cursor();
		else if (tensor instanceof Img)
			tensorCursor = ((Img<IntType>) tensor).cursor();
		else
			throw new IllegalArgumentException("The data of the " + Tensor.class + " has "
					+ "to be an instance of " + Img.class + " or " + IntervalView.class);
		long flatSize = 1;
		for (long dd : tensor.dimensionsAsLongArray()) { flatSize *= dd;}
		int[] flatArr = new int[(int) flatSize];
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	int val = tensorCursor.get().getInteger();
        	flatArr[flatPos] = val;
		}
		int[] shape = IntStream.range(0, tensor.numDimensions()).map(i -> (int) tensor.dimension(i)).toArray();
		NDArray<int[]> nd = new NDArray<int[]>(flatArr, shape);
	 	return nd;
    }

    /**
     * Builds a {@link NDArray} from a unsigned float-typed {@link RandomAccessibleInterval}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The NDArray built from the tensor of type {@link DataType#FLOAT}.
     */
    private static NDArray<float[]> buildFromTensorFloat(RandomAccessibleInterval<FloatType> tensor)
    {
    	long[] tensorShape = tensor.dimensionsAsLongArray();
    	Cursor<FloatType> tensorCursor;
		if (tensor instanceof IntervalView)
			tensorCursor = ((IntervalView<FloatType>) tensor).cursor();
		else if (tensor instanceof Img)
			tensorCursor = ((Img<FloatType>) tensor).cursor();
		else
			throw new IllegalArgumentException("The data of the " + Tensor.class + " has "
					+ "to be an instance of " + Img.class + " or " + IntervalView.class);
		long flatSize = 1;
		for (long dd : tensor.dimensionsAsLongArray()) { flatSize *= dd;}
		float[] flatArr = new float[(int) flatSize];
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	float val = tensorCursor.get().getRealFloat();
        	flatArr[flatPos] = val;
		}
		int[] shape = IntStream.range(0, tensor.numDimensions()).map(i -> (int) tensor.dimension(i)).toArray();
		NDArray<float[]> nd = new NDArray<float[]>(flatArr, shape);
	 	return nd;
    }

    /**
     * Builds a {@link NDArray} from a unsigned double-typed {@link RandomAccessibleInterval}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The NDArray built from the tensor of type {@link DataType#DOUBLE}.
     */
    private static NDArray<double[]> buildFromTensorDouble(RandomAccessibleInterval<DoubleType> tensor)
    {
    	long[] tensorShape = tensor.dimensionsAsLongArray();
    	Cursor<DoubleType> tensorCursor;
		if (tensor instanceof IntervalView)
			tensorCursor = ((IntervalView<DoubleType>) tensor).cursor();
		else if (tensor instanceof Img)
			tensorCursor = ((Img<DoubleType>) tensor).cursor();
		else
			throw new IllegalArgumentException("The data of the " + Tensor.class + " has "
					+ "to be an instance of " + Img.class + " or " + IntervalView.class);
		long flatSize = 1;
		for (long dd : tensor.dimensionsAsLongArray()) { flatSize *= dd;}
		double[] flatArr = new double[(int) flatSize];
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	double val = tensorCursor.get().getRealDouble();
        	flatArr[flatPos] = val;
		}
		int[] shape = IntStream.range(0, tensor.numDimensions()).map(i -> (int) tensor.dimension(i)).toArray();
		NDArray<double[]> nd = new NDArray<double[]>(flatArr, shape);
	 	return nd;
    }

    /**
     * Builds a {@link NDArray} from a unsigned double-typed {@link RandomAccessibleInterval}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The NDArray built from the tensor of type {@link DataType#DOUBLE}.
     */
    private static NDArray<long[]> buildFromTensorLong(RandomAccessibleInterval<LongType> tensor)
    {
    	long[] tensorShape = tensor.dimensionsAsLongArray();
    	Cursor<LongType> tensorCursor;
		if (tensor instanceof IntervalView)
			tensorCursor = ((IntervalView<LongType>) tensor).cursor();
		else if (tensor instanceof Img)
			tensorCursor = ((Img<LongType>) tensor).cursor();
		else
			throw new IllegalArgumentException("The data of the " + Tensor.class + " has "
					+ "to be an instance of " + Img.class + " or " + IntervalView.class);
		long flatSize = 1;
		for (long dd : tensor.dimensionsAsLongArray()) { flatSize *= dd;}
		long[] flatArr = new long[(int) flatSize];
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	long val = tensorCursor.get().getLong();
        	flatArr[flatPos] = val;
		}
		int[] shape = IntStream.range(0, tensor.numDimensions()).map(i -> (int) tensor.dimension(i)).toArray();
		NDArray<long[]> nd = new NDArray<long[]>(flatArr, shape);
	 	return nd;
    }

}
