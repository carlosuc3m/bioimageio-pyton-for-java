package org.bioimageanalysis.icy.deeplearning.python.transformations;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;

public class BioimageioPythonTransformations {
	
	private String name;
	private String transformationObjectKey;
	private Map<String, Object> kwargs;
	
	private static String tensorNameKey = "tensor_name";
	
	private static List<String> binarizeAllArgs = Arrays.stream(new String[] {"tensor_name", "threshold", "mode"}).collect(Collectors.toList());
	private static List<String> binarizeCompulsoryArgs = Arrays.stream(new String[] {"tensor_name", "threshold"}).collect(Collectors.toList());
	
	private static List<String> clipAllArgs = Arrays.stream(new String[] {"tensor_name", "min", "max", "mode"}).collect(Collectors.toList());
	private static List<String> clipCompulsoryArgs = Arrays.stream(new String[] {"tensor_name", "min", "max"}).collect(Collectors.toList());
	
	private static List<String> scaleLinearAllArgs = Arrays.stream(new String[] {"tensor_name", "mode", "gain", "offset", "axes"}).collect(Collectors.toList());
	private static List<String> scaleLinearCompulsoryArgs = 
			Arrays.stream(new String[] {"tensor_name", "gain", "offset"}).collect(Collectors.toList());
	
	private static List<String> scaleMeanVarianceAllArgs = Arrays.stream(new String[] {"tensor_name", "mode"}).collect(Collectors.toList());
	private static List<String> scaleMeanVarianceCompulsoryArgs = Arrays.stream(new String[] {"tensor_name"}).collect(Collectors.toList());
	
	private static List<String> scaleRangeAllArgs = 
			Arrays.stream(new String[] {"tensor_name", "mode", "axes", "min_percentile", "max_percentile", "reference_tensor"}).collect(Collectors.toList());
	private static List<String> scaleRangeCompulsoryArgs = 
			Arrays.stream(new String[] {"tensor_name", "min_percentile", "max_percentile"}).collect(Collectors.toList());
	
	private static List<String> sigmoidAllArgs = Arrays.stream(new String[] {"tensor_name", "mode"}).collect(Collectors.toList());
	private static List<String> sigmoidCompulsoryArgs = Arrays.stream(new String[] {"tensor_name"}).collect(Collectors.toList());
	
	private static List<String> zeroMeanUnitVarianceAllArgs = Arrays.stream(new String[] {"tensor_name", "mode", "mean", "std", "axes"}).collect(Collectors.toList());
	private static List<String> zeroMeanUnitVarianceCompulsoryArgs = Arrays.stream(new String[] {"tensor_name"}).collect(Collectors.toList());
	
	private BioimageioPythonTransformations(Map<String, Object> transformation) {
		this.name = (String) transformation.get("name");
		this.kwargs = (Map<String, Object>) transformation.get("kwargs");
		checkArguments();
	}
	
	public static BioimageioPythonTransformations definePythonBioImageIoTransformation(Map<String, Object> transformation) {
		return new BioimageioPythonTransformations(transformation);
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
	
	private void checkArgumentInListOrThrow(List<String> allArgs, List<String> compulsoryArgs ) {
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
			command += this.getTransformationObjectName() + " = Binarize("
					+ getArgumentsString() + ")" + System.lineSeparator();
		} else if (name.equals("clip")) {
			command += "from bioimageio.core.prediction_pipeline._processing import Clip" + System.lineSeparator();
			command += this.getTransformationObjectName() + " = Clip(" + getArgumentsString() + ")";
		} else if (name.equals("scale_linear")) {
			command += "from bioimageio.core.prediction_pipeline._processing import ScaleLinear" + System.lineSeparator();
			command += this.getTransformationObjectName() + " = ScaleLinear(" + getArgumentsString() + ")";
		} else if (name.equals("scale_mean_variance")) {
			// TODO finish when the Python transformation
			command += "from bioimageio.core.prediction_pipeline._processing import ScaleMeanVariance" + System.lineSeparator();
			command += this.getTransformationObjectName() + " = ScaleMeanVariance(" + getArgumentsString() + ")";
		} else if (name.equals("scale_range")) {
			// TODO finish when the Python transformation
			command += "from bioimageio.core.prediction_pipeline._processing import ScaleRange" + System.lineSeparator();
			command += this.getTransformationObjectName() + " = ScaleRange(" + getArgumentsString() + ")";
		} else if (name.equals("sigmoid")) {
			command += "from bioimageio.core.prediction_pipeline._processing import Sigmoid" + System.lineSeparator();
			command += this.getTransformationObjectName() + " = Sigmoid(" + getArgumentsString() + ")";
		} else if (name.equals("zero_mean_unit_variance")) {
			command += "from bioimageio.core.prediction_pipeline._processing import ZeroMeanUnitVariance" + System.lineSeparator();
			command += this.getTransformationObjectName() + " = ZeroMeanUnitVariance(" + getArgumentsString() + ")";
		}
		command += System.lineSeparator();
		command += addComputedMeasures(tensorName);
		return command;
	}
	
	private String addComputedMeasures(String tensorName) {
		String command = "";
		command += "required = " + this.getTransformationObjectName() + ".get_required_measures()" + System.lineSeparator();
		command += "computed = compute_measures(required, sample={\"" + this.kwargs.get(tensorNameKey) +
				"\": " + tensorName + "})" + System.lineSeparator();
		command += this.getTransformationObjectName() + ".set_computed_measures(computed)" + System.lineSeparator();
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
