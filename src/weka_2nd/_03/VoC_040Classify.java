package weka_2nd._03;


import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka_2nd.MoreWekaCommon;

public class VoC_040Classify {

	private Classifier model = null;
		
	public VoC_040Classify (Classifier model){
		this.model = model;
	}
	
	public Object loadModel(String modelName) throws Exception{
		return weka.core.SerializationHelper.read(MoreWekaCommon.modelPath +"textMining\\voc_"
				                                  +modelName);
	}
	
	public void useModel (Instances test) throws Exception{
			
		Classifier model = (Classifier)loadModel(MoreWekaCommon.getModelName(this.model));
		System.out.println("Classifier model : " + MoreWekaCommon.getModelName(model));
		test.setClassIndex(test.numAttributes()-1);
		
		System.out.println(test.relationName());
		
		// 유사 신규데이터 (단어)의 기존모델 일치여부 판단
		int inCorrectCnt = 0;
		for(int x=0 ; x < test.numInstances(); x++){	
			
			if(test.get(x).classValue() != model.classifyInstance(test.get(x))){
				inCorrectCnt ++;
				System.out.println(
						(x+1) + " st : " +
						" test : model = " + 
						test.get(x).classValue() + 
						" : " + 
						model.classifyInstance(test.get(x)) +
						(test.get(x).classValue() == model.classifyInstance(test.get(x))? " 일치" : " 불일치") + 
						" : " + test.get(x)
						);			
			}
		}	
		System.out.println("불일치 건수 : " + inCorrectCnt + 
				          " , 불일치율 : " + String.format("%.1f", (double)inCorrectCnt/(double)test.numInstances() * 100) + " %");
	}

	public void useModel (Instances train, Instances test) throws Exception{
		
		Classifier model = (Classifier)loadModel(MoreWekaCommon.getModelName(this.model));
		System.out.println("Classifier model : " + MoreWekaCommon.getModelName(model));
		
		Evaluation eval = new Evaluation(train);
		eval.evaluateModel(model, test); 
		
		System.out.println(eval.predictions());
	}
	

}
