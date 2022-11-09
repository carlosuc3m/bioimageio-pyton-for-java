package org.bioimageanalysis.icy.deeplearning.python;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.bioimageanalysis.icy.deeplearning.python.tensor.PythonTensor;
import org.bioimageanalysis.icy.deeplearning.python.transformations.BioimageioPythonTransformations;
import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;
import org.bioimageanalysis.icy.jep.exec.PythonExec;
import org.bioimageanalysis.icy.jep.install.system.Log;
import org.bioimageanalysis.icy.jep.utils.JepUtils;

import jep.Interpreter;
import jep.JepConfig;
import jep.NDArray;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

/*
 * @author Carlos Garcia Lopez de Haro
 */
public class BioImageIoPython implements AutoCloseable {
	
	private PythonExec pythonExec;
	private Interpreter interp;
	private boolean isInstalled = false;
	private String version;
	private List<String> instantiatedTransformations = new ArrayList<String>();
	private String instantiatedNpArray;
	private String instantiatedTensor;
	private static String BIOIMAGE_IO_PACKAGE_NAME = "bioimageio.core";
	private static String DEFAULT_BIOIMAGEIO_VERSION = "0.5.6";
	
	private BioImageIoPython(PythonExec pythonExec) throws IOException, InterruptedException {
		this.pythonExec = pythonExec;
		this.interp = pythonExec.getInterpreter();
		checkInstalled();
		if (!isInstalled)
			install();
		importGenericModules();
	}

	public static BioImageIoPython activate(String pythonHome, String jepPath) 
												throws IllegalArgumentException, IOException, InterruptedException {
		return new BioImageIoPython(PythonExec.build(pythonHome, jepPath));
	}

	public static BioImageIoPython activate(String pythonHome, String jepPath, JepConfig jepConfig) 
												throws IllegalArgumentException, IOException, InterruptedException {
		return new BioImageIoPython(PythonExec.build(pythonHome, jepPath, jepConfig));
	}

	public static BioImageIoPython activate(PythonExec pythonExec) throws IOException, InterruptedException {
		return new BioImageIoPython(pythonExec);
	}
	
	public static void main(String[] args) throws IllegalArgumentException, IOException, InterruptedException {
		activate(JepUtils.getInstance().getPythonInstance());
	}
	
	public void install() throws IOException, InterruptedException {
		install(DEFAULT_BIOIMAGEIO_VERSION);
	}
	
	public void install(String version) throws IOException, InterruptedException {
		pythonExec.installPythonPackage(BIOIMAGE_IO_PACKAGE_NAME + "==" + version);
		this.version = version;
		isInstalled = true;
	}
	
	public boolean checkInstalled() {
		if (isInstalled)
			return isInstalled;
		isInstalled = pythonExec.checkInstalled(BIOIMAGE_IO_PACKAGE_NAME);
		return isInstalled;		
	}
	
	private void importGenericModules() {
		if (interp == null)
			throw new IllegalArgumentException("There should be a 'SharedInterpreter' open.");
		System.out.println(Log.getCurrentTime() + " -- Importing required packages for " + BIOIMAGE_IO_PACKAGE_NAME);
        interp.exec("import numpy as np" + System.lineSeparator());
        interp.exec("import xarray as xr" + System.lineSeparator());
        interp.exec("from bioimageio.core.prediction_pipeline._measure_groups import compute_measures" + System.lineSeparator());
        interp.exec("from bioimageio.core.prediction_pipeline._utils import PER_SAMPLE, FIXED, PER_DATASET" + System.lineSeparator());
	}
	
	public void instantiatePythonTransformationObject(Map<String, Object> transformationMap, String tensorName) {
		BioimageioPythonTransformations bioimageioPythonTransfomation = BioimageioPythonTransformations.definePythonBioImageIoTransformation(transformationMap);
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
		interp.exec("tensor_axes_order = " + this.instantiatedTensor +".dims");
		List<String> tensorDimsArr = (List<String>) interp.getValue("tensor_axes_order");
		String axesOrder = "";
		for (String ii : tensorDimsArr)
			axesOrder += ii;
		interp.exec("tensor_np_array = " + instantiatedTensor +".data");
		NDArray<?> jepArray = interp.getValue("tensor_np_array", NDArray.class);
		return PythonTensor.build(javaTensor.getName(), axesOrder, jepArray);
	}
	
	public < T extends RealType< T > & NativeType< T > > Tensor<T> applyTransformationToTensorInPython(Map<String, Object> transformationMap, 
																						Tensor<T> javaTensor) {
		sendTensorToInterpreter(javaTensor);		
		instantiatePythonTransformationObject(transformationMap, instantiatedTensor);
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
	
	public String getInstalledVersion() {
		return version;
	}
	
	public static String getDefaultVersion() {
		return DEFAULT_BIOIMAGEIO_VERSION;
	}
}
