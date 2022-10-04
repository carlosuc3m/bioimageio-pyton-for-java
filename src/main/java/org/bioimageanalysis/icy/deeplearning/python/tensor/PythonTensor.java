package org.bioimageanalysis.icy.deeplearning.python.tensor;

import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;

import jep.NDArray;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;

/**
 * Class that converts Java tensors into objects that can easily be translated into Python BioImage.io
 * tensors using JEP
 * @author Carlos Garcia Lopez de Haro
 *
 */

public class PythonTensor {
	/**
	 * Name of the tensor
	 */
	private String name;
	/**
	 * Axes order of the tensor
	 */
	private String axesOrder;
	/**
	 * Data contained in the tensor
	 */
	private NDArray<?> data;
	/**
	 * Data type of the tensor
	 */
	private String dataType;
	/**
	 * Shape of the tensor
	 */
	private int[] shape;
	/**
	 * Unique identifier used to denominate the Numpy array of this tensor in the Python scope
	 */
	private String npArrayKey;
	/**
	 * Unique identifier used to denominate the tensor of this tensor in the Python scope
	 */
	private String tensorKey;
	
	/**
	 * Constructor to create a tensor that can easily be converted into a Python BioImage.io tensor
	 * @param name
	 * 	name of the tensor
	 * @param axesOrder
	 * 	axes order of the tensor
	 * @param nd
	 * 	data of the tensor as a JEP NDArray
	 */
	private PythonTensor(String name, String axesOrder, NDArray< ? > nd) {
		this.name = name;
		this.axesOrder = axesOrder;
		this.data = nd;
		this.shape = nd.getDimensions();
	}
	
	/**
	 * Create a tensor that can easily be converted into a Python BioImage.io tensor
	 * @param name
	 * 	name of the tensor
	 * @param axesOrder
	 * 	axes order of the tensor
	 * @param nd
	 * 	data of the tensor as a JEP NDArray
	 */
	public static PythonTensor build(String name, String axesOrder, NDArray< ? > nd) {
		return new PythonTensor(name, axesOrder, nd);
	}
	
	/**
	 * 
	 * @return the name of the tensor
	 */
	public String getName() {
		return name;
	}
	
	/**
	 * 
	 * @return the axes order of the tensor
	 */
	public String getAxesOrder() {
		return axesOrder;
	}
	
	/**
	 * 
	 * @return the {@link NDArray} that contains the data of the tensor
	 */
	public NDArray<?> getData(){
		return data;
	}
	
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
			float[] arr = RaiArrayUtils.floatArray((RandomAccessibleInterval<FloatType>) data);
			NDArray<float[]> nd = new NDArray<float[]>(arr, javaTensor.getShape());
			return new PythonTensor(javaTensor.getName(), javaTensor.getAxesOrderString(), nd);
		} else if (dt instanceof IntType) {
			int[] arr = RaiArrayUtils.intArray((RandomAccessibleInterval<IntType>) data);
			NDArray<int[]> nd = new NDArray<int[]>(arr, javaTensor.getShape());
			return new PythonTensor(javaTensor.getName(), javaTensor.getAxesOrderString(), nd);
		} else if (dt instanceof DoubleType) {
			double[] arr = RaiArrayUtils.doubleArray((RandomAccessibleInterval<DoubleType>) data);
			NDArray<double[]> nd = new NDArray<double[]>(arr, javaTensor.getShape());
			return new PythonTensor(javaTensor.getName(), javaTensor.getAxesOrderString(), nd);
		} else if (dt instanceof LongType) {
			long[] arr = RaiArrayUtils.longArray((RandomAccessibleInterval<LongType>) data);
			NDArray<?> nd = new NDArray<long[]>(arr, javaTensor.getShape());
			return new PythonTensor(javaTensor.getName(), javaTensor.getAxesOrderString(), nd);
			
		} else if (dt instanceof ByteType) {
			byte[] arr = RaiArrayUtils.byteArray((RandomAccessibleInterval<ByteType>) data);
			NDArray<byte[]> nd = new NDArray<byte[]>(arr, javaTensor.getShape());
			return new PythonTensor(javaTensor.getName(), javaTensor.getAxesOrderString(), nd);
		} else {
			throw new IllegalArgumentException("Conversion into Python of tensors with dsta type: '" + dt.getClass()+ "' "
					+ "is not supported.");
		}
	}
	
	public < T extends RealType< T > & NativeType< T > > Tensor<T> toJava() {
		long[] longShape = new long[data.getDimensions().length];
		for (int i = 0; i < longShape.length; i ++)
			longShape[i] = data.getDimensions()[i];
		if (data.getData() instanceof int[]) {
			return (Tensor<T>) Tensor.build(name, axesOrder, ArrayImgs.ints((int[])data.getData(), longShape));
		} else if (data.getData() instanceof float[]) {
			return (Tensor<T>) Tensor.build(name, axesOrder, ArrayImgs.floats((float[])data.getData(), longShape));
		} else if (data.getData() instanceof double[]) {
			return (Tensor<T>) Tensor.build(name, axesOrder, ArrayImgs.doubles((double[])data.getData(), longShape));
		} else if (data.getData() instanceof long[]) {
			return (Tensor<T>) Tensor.build(name, axesOrder, ArrayImgs.longs((long[])data.getData(), longShape));
		} else if (data.getData() instanceof byte[]) {
			return (Tensor<T>) Tensor.build(name, axesOrder, ArrayImgs.bytes((byte[])data.getData(), longShape));
		} else {
			throw new IllegalArgumentException("");
		}
	}
	
	/**
	 * Method to create a String command that would create in Python a BioImage.io tensor.
	 * And example result would be the following:
	 *  - "data = xr.DataArray(np_data, dims=axes)""
	 * @return
	 */
	public String createCommandToBuildPythonBioiamgeIoTensor() {
		// First check if xr is imported and if it is not import it
		String command = "if \"xarray\" not in sys.modules:" + System.lineSeparator();
		command += "\timport xarray as xr" + System.lineSeparator();
		command += "if \"numpy\" not in sys.modules:" + System.lineSeparator();
		command += "\timport numpy as np" + System.lineSeparator();
		command += this.getTensorName() + " = xr.DataArray(" + this.getNpArrayVarName()
				+ ", dims=(";
		for (String ax : this.getAxesOrder().split(""))
			command += "\"" + ax + "\",";
		command += "))";
		return command;
	}
	
	/**
	 * Unique name that will be used to denominate the Numpy array equivalent to this tensor data in the Python
	 * scope
	 * @return unique name used to denominate the Numpy array of this tensor in the Python scope
	 */
	public String getNpArrayVarName() {
		if (npArrayKey == null)
			npArrayKey = name + "_np_array_" + System.currentTimeMillis();
		return npArrayKey;
	}

	/**
	 * Unique name that will be used to denominate the tensor array equivalent to this tensor data in the Python
	 * scope
	 * @return unique name used to denominate the tensor array of this tensor in the Python scope
	 */
	public String getTensorName() {
		if (tensorKey == null)
			tensorKey = name + "_tensor_" + System.currentTimeMillis();
		return tensorKey;
	}

}
