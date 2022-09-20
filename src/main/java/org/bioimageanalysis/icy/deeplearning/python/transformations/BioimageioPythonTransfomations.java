package org.bioimageanalysis.icy.deeplearning.python.transformations;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map.Entry;

public class BioimageioPythonTransfomations {
	
	private String name;
	private String transformationObjectKey;
	private HashMap<String, Object> kwargs;
	
	private static ArrayList<String> binarizeAllArgs = (ArrayList<String>) Arrays.asList(new String[] {"tensor_name", "threshold", "mode"});
	private static ArrayList<String> binarizeCompulsoryArgs = (ArrayList<String>) Arrays.asList(new String[] {"tensor_name", "threshold"});
	
	private static ArrayList<String> clipAllArgs = (ArrayList<String>) Arrays.asList(new String[] {"tensor_name", "min", "max", "mode"});
	private static ArrayList<String> clipCompulsoryArgs = (ArrayList<String>) Arrays.asList(new String[] {"tensor_name", "min", "max"});
	
	private static ArrayList<String> scaleLinearAllArgs = (ArrayList<String>) Arrays.asList(new String[] {"tensor_name", "mode", "gain", "offset", "axes"});
	private static ArrayList<String> scaleLinearCompulsoryArgs = 
			(ArrayList<String>) Arrays.asList(new String[] {"tensor_name", "gain", "offset"});
	
	private static ArrayList<String> scaleMeanVarianceAllArgs = (ArrayList<String>) Arrays.asList(new String[] {"tensor_name", "mode"});
	private static ArrayList<String> scaleMeanVarianceCompulsoryArgs = (ArrayList<String>) Arrays.asList(new String[] {"tensor_name"});
	
	private static ArrayList<String> scaleRangeAllArgs = 
			(ArrayList<String>) Arrays.asList(new String[] {"tensor_name", "mode", "axes", "min_percentile", "max_percentile", "reference_tensor"});
	private static ArrayList<String> scaleRangeCompulsoryArgs = 
			(ArrayList<String>) Arrays.asList(new String[] {"tensor_name", "min_percentile", "max_percentile"});
	
	private static ArrayList<String> sigmoidAllArgs = (ArrayList<String>) Arrays.asList(new String[] {"tensor_name", "mode"});
	private static ArrayList<String> sigmoidCompulsoryArgs = (ArrayList<String>) Arrays.asList(new String[] {"tensor_name"});
	
	private static ArrayList<String> zeroMeanUnitVarianceAllArgs = (ArrayList<String>) Arrays.asList(new String[] {"tensor_name", "mode", "mean", "std", "axes"});
	private static ArrayList<String> zeroMeanUnitVarianceCompulsoryArgs = (ArrayList<String>) Arrays.asList(new String[] {"tensor_name"});
	
	private BioimageioPythonTransfomations(HashMap<String, Object> transformation) {
		this.name = (String) transformation.get("name");
		this.kwargs = (HashMap<String, Object>) transformation.get("kwargs");
		checkArguments();
	}
	
	public static BioimageioPythonTransfomations definePythonBioImageIoTransformation(HashMap<String, Object> transformation) {
		return new BioimageioPythonTransfomations(transformation);
	}
	
	private void checkIfValueIsOfValidType(Entry<String, Object> entry) {
		Object val = entry.getValue();
		if (!(val instanceof String) && !(val instanceof Number) 
				&& !(val instanceof ArrayList) && !(val instanceof double[])
				&& !(val instanceof long[]) && !(val instanceof int[])
				&& !(val instanceof float[]) && !(val instanceof byte[])) {
			throw new IllegalArgumentException("The value of argument '"+ entry.getKey() 
						+ "' is not a valid object (number, String or an array of the previous).");
		}
	}
	
	private void checkArgumentInListOrThrow(ArrayList<String> allArgs, ArrayList<String> compulsoryArgs ) {
		int n = 0;
		for (Entry<String, Object> entry : kwargs.entrySet()) {
			if (!allArgs.contains(entry.getKey()))
				throw new IllegalArgumentException("The argument '" + entry.getKey() + "' is not a valid argument of the '" + name
						+ "' transformation (" + allArgs + ")");
			else if (compulsoryArgs.contains(entry.getKey()))
				n += 1;
			checkIfValueIsOfValidType(entry);
		}
		
		if (n != compulsoryArgs.size())
			throw new IllegalArgumentException("One or more of the compulsory arguments: " + compulsoryArgs 
					+ " for the BioImage.io transformation '" + name + "' is missing.");
	}
	
	private void checkArguments() {
		if (name.equals("binarize")) {
			checkArgumentInListOrThrow(binarizeAllArgs, binarizeCompulsoryArgs);
		} else if (name.equals("clip")) {
			checkArgumentInListOrThrow(clipAllArgs, clipCompulsoryArgs);
		} else if (name.equals("scale_linear")) {
			checkArgumentInListOrThrow(scaleLinearAllArgs, scaleLinearCompulsoryArgs);
		} else if (name.equals("scale_mean_variance")) {
			// TODO finish when the Python transformation
			checkArgumentInListOrThrow(scaleMeanVarianceAllArgs, scaleMeanVarianceCompulsoryArgs);
		} else if (name.equals("scale_range")) {
			checkArgumentInListOrThrow(scaleRangeAllArgs, scaleRangeCompulsoryArgs);
		} else if (name.equals("sigmoid")) {
			checkArgumentInListOrThrow(sigmoidAllArgs, sigmoidCompulsoryArgs);
		} else if (name.equals("zero_mean_unit_variance")) {
			checkArgumentInListOrThrow(zeroMeanUnitVarianceAllArgs, zeroMeanUnitVarianceCompulsoryArgs);
		}
	}
	
	/**
	 * Method that creates a String of a Python command that can be used to instantiate a BioImage.io
	 * transformation in Python.
	 * For example, some possible results could be:
	 *  - "from bioimageio.core.prediction_pipeline._processing import ScaleLinear\n"
	 *  	+ "processing = ScaleLinear("data_name", offset=[1, 2, 42], gain=[1, 2, 3], axes="yx")"
	 *  - "from bioimageio.core.prediction_pipeline._processing import ZeroMeanUnitVariance\n"
	 *  	+ "processing = ZeroMeanUnitVariance("data_name", mode=PER_SAMPLE)"
	 *  
	 * @param tensorName
	 * 	name of the tensor to which the transformation is going to be applied
	 * @return
	 */
	public String stringToInstantiatePythonTransformation(String tensorName) {
		String command = "";
		if (name.equals("binarize")) {
			command += "from bioimageio.core.prediction_pipeline._processing import Binarize" + System.lineSeparator();
			command += this.getTransformationObjectName() + " = Binarize(tensor_name=" + tensorName
					+ ", " + getArgumentsString() + ")";
		} else if (name.equals("clip")) {
			command += "from bioimageio.core.prediction_pipeline._processing import Clip" + System.lineSeparator();
			command += this.getTransformationObjectName() + " = Clip(tensor_name=" + tensorName
					+ ", " + getArgumentsString() + ")";
		} else if (name.equals("scale_linear")) {
			command += "from bioimageio.core.prediction_pipeline._processing import ScaleLinear" + System.lineSeparator();
			command += this.getTransformationObjectName() + " = ScaleLinear(tensor_name=" + tensorName
					+ ", " + getArgumentsString() + ")";
		} else if (name.equals("scale_mean_variance")) {
			// TODO finish when the Python transformation
			command += "from bioimageio.core.prediction_pipeline._processing import ScaleMeanVariance" + System.lineSeparator();
			command += this.getTransformationObjectName() + " = ScaleMeanVariance(tensor_name=" + tensorName
					+ ", " + getArgumentsString() + ")";
		} else if (name.equals("scale_range")) {
			// TODO finish when the Python transformation
			command += "from bioimageio.core.prediction_pipeline._processing import ScaleRange" + System.lineSeparator();
			command += this.getTransformationObjectName() + " = ScaleRange(tensor_name=" + tensorName
					+ ", " + getArgumentsString() + ")";
		} else if (name.equals("sigmoid")) {
			command += "from bioimageio.core.prediction_pipeline._processing import Sigmoid" + System.lineSeparator();
			command += this.getTransformationObjectName() + " = Sigmoid(tensor_name=" + tensorName
					+ ", " + getArgumentsString() + ")";
		} else if (name.equals("zero_mean_unit_variance")) {
			command += "from bioimageio.core.prediction_pipeline._processing import ZeroMeanUnitVariance" + System.lineSeparator();
			command += this.getTransformationObjectName() + " = ZeroMeanUnitVariance(tensor_name=" + tensorName
					+ ", " + getArgumentsString() + ")";
		}
		
		return command;
	}
	
	private String getArgumentsString() {
		String command = "";
		for (Entry<String, Object> entry : kwargs.entrySet()) {
			command += entry.getKey() + "=";
			if (entry.getValue() instanceof String) {
				command += "\"" + entry.getValue() + "\", ";
			} else if (entry.getValue() instanceof Number) {
				command += entry.getValue() + ", ";
			} else if (entry.getValue() instanceof ArrayList) {
				command += entry.getValue() + ", ";
			} else if (entry.getValue() instanceof int[]) {
				command += Arrays.toString((int[]) entry.getValue()) + ", ";
			} else if (entry.getValue() instanceof float[]) {
				command += Arrays.toString((float[]) entry.getValue()) + ", ";
			} else if (entry.getValue() instanceof long[]) {
				command += Arrays.toString((long[]) entry.getValue()) + ", ";
			} else if (entry.getValue() instanceof double[]) {
				command += Arrays.toString((double[]) entry.getValue()) + ", ";
			} else if (entry.getValue() instanceof byte[]) {
				command += Arrays.toString((byte[]) entry.getValue()) + ", ";
			}			
		}
		return command;
	}

	/**
	 * Unique name that will be used to denominate the Numpy array equivalent to this tensor data in the Python
	 * scope
	 * @return unique name used to denominate the Numpy array of this tensor in the Python scope
	 */
	public String getTransformationObjectName() {
		if (transformationObjectKey == null)
			transformationObjectKey = name + "_" + System.currentTimeMillis();
		return transformationObjectKey;
	}
}
