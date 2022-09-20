package org.bioimageanalysis.icy.deeplearning.python;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.bioimageanalysis.icy.deeplearning.python.tensor.PythonTensor;
import org.bioimageanalysis.icy.deeplearning.python.transformations.BioimageioPythonTransfomations;
import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;
import org.bioimageanalysis.icy.jep.exec.PythonExec;

import jep.Interpreter;
import jep.JepConfig;
import jep.NDArray;
import jep.SharedInterpreter;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

public class BioImageIoPython implements Closeable {
	
	private PythonExec pythonExec;
	private Interpreter interp;
	private List<String> instantiatedTransformations = new ArrayList<String>();
	private String instantiatedNpArray;
	private String instantiatedTensor;
	
	private BioImageIoPython(PythonExec pythonExec) {
		this.pythonExec = pythonExec;
		this.interp = pythonExec.getInterpreter();
		importGenericModules();
	}

	public static BioImageIoPython activate(String pythonHome, String jepPath) 
												throws IllegalArgumentException, IOException {
		return new BioImageIoPython(PythonExec.build(pythonHome, jepPath));
	}

	public static BioImageIoPython activate(String pythonHome, String jepPath, JepConfig jepConfig) 
												throws IllegalArgumentException, IOException {
		return new BioImageIoPython(PythonExec.build(pythonHome, jepPath, jepConfig));
	}

	public static BioImageIoPython activate(PythonExec pythonExec) {
		return new BioImageIoPython(pythonExec);
	}
	
	private void importGenericModules() {
		if (interp == null)
			throw new IllegalArgumentException("There should be a 'SharedInterpreter' open.");

        interp.exec("import numpy as np" + System.lineSeparator());
        interp.exec("import xarray as xr" + System.lineSeparator());
        interp.exec("from bioimageio.core.prediction_pipeline._measure_groups import compute_measures" + System.lineSeparator());
        interp.exec("from bioimageio.core.prediction_pipeline._utils import PER_SAMPLE" + System.lineSeparator());
	}
	
	public void instantiatePythonTransformationObject(HashMap<String, Object> transformationMap, String tensorName) {
		BioimageioPythonTransfomations bioimageioPythonTransfomation = BioimageioPythonTransfomations.definePythonBioImageIoTransformation(transformationMap);
		String pythonCommand = bioimageioPythonTransfomation.stringToInstantiatePythonTransformation(tensorName);
        interp.exec(pythonCommand);
        instantiatedTransformations.add(bioimageioPythonTransfomation.getTransformationObjectName());
	}
	
	public < T extends RealType< T > & NativeType< T > > void sendTensorToInterpreter(Tensor<T> javaTensor) {
		PythonTensor pythonTensor = PythonTensor.fromJavaTensor(javaTensor);
		interp.set(pythonTensor.getNpArrayVarName(), pythonTensor.getData()); 
		instantiatedNpArray =pythonTensor.getNpArrayVarName();
		String pythonCommand = pythonTensor.createCommandToBuildPythonBioiamgeIoTensor();
        interp.exec(pythonCommand);
        instantiatedTensor = pythonTensor.getTensorName();
		
	}

	public PythonTensor retrieveBioImageIoPythonTensorFromScope(Tensor javaTensor) {
		interp.exec("tensor_axes_order = " + instantiatedNpArray +".dims");
		ArrayList<String> tensorDimsArr = interp.getValue("tensor_axes_order", ArrayList.class);
		String axesOrder = "";
		for (String ii : tensorDimsArr)
			axesOrder += ii;
		interp.exec("tensor_np_array = " + instantiatedNpArray +".data");
		NDArray<?> jepArray = interp.getValue("tensor_np_array", NDArray.class);
		return PythonTensor.build(javaTensor.getName(), axesOrder, jepArray);
	}
	
	public < T extends RealType< T > & NativeType< T > > Tensor<T> applyTransformationToTensorInPython(HashMap<String, Object> transformationMap, 
																						Tensor<T> javaTensor) {
		instantiatePythonTransformationObject(transformationMap, javaTensor.getName());
		sendTensorToInterpreter(javaTensor);		
		executeTransformations();
		PythonTensor result = retrieveBioImageIoPythonTensorFromScope(javaTensor);
		return result.toJava();
	}
	
	public void executeTransformations() {
		String command = "";
		for (String transformation : instantiatedTransformations) {
			command = instantiatedTensor + " = " + transformation + ".apply(" + instantiatedTensor + ")" + System.lineSeparator();
		}
		interp.exec(command);
	}
	
	public PythonExec getPythonExecutor() {
		return this.pythonExec;
	}
	
	public Interpreter getBioImageIoPythonInterpreter() {
		return this.interp;
	}

	public void close() throws IOException {
		if (pythonExec != null )
			pythonExec.close();
		if (interp != null)
			interp.close();
	}
	
	
}