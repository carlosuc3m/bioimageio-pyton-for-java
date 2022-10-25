package org.bioimageanalysis.icy.deeplearning.python.tensor;

import java.util.stream.LongStream;

import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;
import org.bioimageanalysis.icy.deeplearning.utils.IndexingUtils;

import jep.NDArray;
import net.imglib2.Cursor;
import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.cell.CellImgFactory;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;

/**
 * Class that converts Java tensors into objects that can easily be translated into Python BioImage.io
 * tensors using JEP
 * @author Carlos Garcia Lopez de Haro
 *
 */

public class PythonToJavaTensor {
	
	public static < T extends RealType< T > & NativeType< T > > Img<T> build(NDArray<?> data) {
        if (data.getData() instanceof int[] || data.getData() instanceof byte[]) {
            return (Img<T>) buildFromTensorInt((NDArray<int[]>) data);
		} else if (data.getData() instanceof float[]) {
            return (Img<T>) buildFromTensorFloat((NDArray<float[]>) data);
		} else if (data.getData() instanceof double[]) {
            return (Img<T>) buildFromTensorDouble((NDArray<double[]>) data);
		} else if (data.getData() instanceof long[]) {
            return (Img<T>) buildFromTensorLong((NDArray<long[]>) data);
		} else if (data.getData() instanceof byte[]) {
            return (Img<T>) buildFromTensorByte((NDArray<byte[]>) data);
		} else {
			throw new IllegalArgumentException("Invalid data type of Python tensor:" + data.getData().getClass());
		}
	}

    /**
     * Builds a {@link Img} from a unsigned byte-typed {@link NDArray}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#UBYTE}.
     */
    private static Img<ByteType> buildFromTensorByte(NDArray<byte[]> tensor)
    {
    	int[] tensorIntShape = tensor.getDimensions();
    	long[] tensorShape = LongStream.range(0, tensor.getDimensions().length)
    			.map(i -> tensorIntShape[(int) i]).toArray();
    	final ImgFactory< ByteType > factory = new CellImgFactory<>( new ByteType(), 5 );
        final Img< ByteType > outputImg = (Img<ByteType>) factory.create(tensorShape);
    	Cursor<ByteType> tensorCursor= outputImg.cursor();
		byte[] flatArr = tensor.getData();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	byte val = flatArr[flatPos];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
	}

    /**
     * Builds a {@link Img} from a unsigned integer-typed {@link NDArray}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#INT}.
     */
    private static Img<IntType> buildFromTensorInt(NDArray<int[]> tensor)
    {
    	int[] tensorIntShape = tensor.getDimensions();
    	long[] tensorShape = LongStream.range(0, tensor.getDimensions().length)
    			.map(i -> tensorIntShape[(int) i]).toArray();
    	final ImgFactory< IntType > factory = new CellImgFactory<>( new IntType(), 5 );
        final Img< IntType > outputImg = (Img<IntType>) factory.create(tensorShape);
    	Cursor<IntType> tensorCursor= outputImg.cursor();
		int[] flatArr = tensor.getData();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	int val = flatArr[flatPos];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

    /**
     * Builds a {@link Img} from a unsigned float-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#FLOAT}.
     */
    private static Img<FloatType> buildFromTensorFloat(NDArray<float[]> tensor)
    {
    	int[] tensorIntShape = tensor.getDimensions();
    	long[] tensorShape = LongStream.range(0, tensor.getDimensions().length)
    			.map(i -> tensorIntShape[(int) i]).toArray();
    	final ImgFactory< FloatType > factory = new CellImgFactory<>( new FloatType(), 5 );
        final Img< FloatType > outputImg = (Img<FloatType>) factory.create(tensorShape);
    	Cursor<FloatType> tensorCursor= outputImg.cursor();
		float[] flatArr = tensor.getData();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	float val = flatArr[flatPos];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

    /**
     * Builds a {@link Img} from a unsigned double-typed {@link NDArray}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#DOUBLE}.
     */
    private static Img<DoubleType> buildFromTensorDouble(NDArray<double[]> tensor)
    {
    	int[] tensorIntShape = tensor.getDimensions();
    	long[] tensorShape = LongStream.range(0, tensor.getDimensions().length)
    			.map(i -> tensorIntShape[(int) i]).toArray();
    	final ImgFactory< DoubleType > factory = new CellImgFactory<>( new DoubleType(), 5 );
        final Img< DoubleType > outputImg = (Img<DoubleType>) factory.create(tensorShape);
    	Cursor<DoubleType> tensorCursor= outputImg.cursor();
		double[] flatArr = tensor.getData();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	double val = flatArr[flatPos];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

    /**
     * Builds a {@link Img} from a unsigned double-typed {@link NDArray}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#DOUBLE}.
     */
    private static Img<LongType> buildFromTensorLong(NDArray<long[]> tensor)
    {
    	int[] tensorIntShape = tensor.getDimensions();
    	long[] tensorShape = LongStream.range(0, tensor.getDimensions().length)
    			.map(i -> tensorIntShape[(int) i]).toArray();
    	final ImgFactory< LongType > factory = new CellImgFactory<>( new LongType(), 5 );
        final Img< LongType > outputImg = (Img<LongType>) factory.create(tensorShape);
    	Cursor<LongType> tensorCursor= outputImg.cursor();
		long[] flatArr = tensor.getData();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	long val = flatArr[flatPos];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }
}
