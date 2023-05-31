package weka_2nd._01;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka_2nd.MoreWekaCommon;

public class Titanic_04Classify {

	private Classifier model = null;
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}
	
	public Titanic_04Classify (Classifier model){
		System.out.println("Classifier model : " + MoreWekaCommon.getModelName(model));
		this.model = model;
	}
	
	public Object loadModel(String modelName) throws Exception{
		return weka.core.SerializationHelper.read(MoreWekaCommon.modelPath +"titanic\\titanic_"
				                                  +modelName);
	}
	
	public void useModel (Instances data) throws Exception{
		
		Classifier model = (Classifier)loadModel(MoreWekaCommon.getModelName(this.model));
		data.setClassIndex(data.numAttributes()-1);
		for(int x=0 ; x < data.numInstances(); x++){
			Instance inst = data.get(x);
			if (data.get(x).classValue() != model.classifyInstance(inst))
				System.out.println(
						(x+1) + 
						" : " +
						data.get(x).toString(data.numAttributes()-1) + 
						" : " + 
						data.get(x).classValue() + 
						" vs " + 
						model.classifyInstance(inst));			
		}	
	}
}
