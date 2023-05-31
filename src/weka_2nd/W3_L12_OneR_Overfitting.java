package weka_2nd;

import java.io.*;
import java.util.Random;
import weka.classifiers.*;
import weka.classifiers.rules.*;
import weka.classifiers.trees.J48;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class W3_L12_OneR_Overfitting {
	
	public W3_L12_OneR_Overfitting(){
		try{
			Classifier model = new J48();
			model.buildClassifier(null);
		}catch(Exception e){
			System.out.println("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
		}
	}
	
	public static void main(String args[]) throws Exception{
		W3_L12_OneR_Overfitting obj = new W3_L12_OneR_Overfitting();
		
		/** weather.numeric ȣ�� **/
		System.out.println("weather.numeric : ");
		obj.weatherNumericHoldOutOneR(false,6);  // ���� ������,  minBucketSize = 6
		obj.weatherNumericHoldOutOneR(true,6);   // ���� ����,   minBucketSize = 6
		obj.weatherNumericHoldOutOneR(true,1);   // ���� ����,   minBucketSize = 1
		
		System.out.println("");
		
		/** diabete ȣ�� **/
		System.out.println("diabete : ");
		obj.diabeteCrossValidationOneR(new ZeroR(),false,6); // zeroR,  crossValidate, minBucketSize = 6
		obj.diabeteCrossValidationOneR(new OneR() ,false,6); // OneR,    crossValidate, minBucketSize = 6
		obj.diabeteCrossValidationOneR(new OneR() ,false,1); // OneR,    crossValidate, minBucketSize = 1
		obj.diabeteCrossValidationOneR(new OneR() ,true,1);  // OneR, Use training set, minBucketSize = 1
	}

	public void weatherNumericHoldOutOneR(boolean isRemove, int minBucketSize) throws Exception{
		int seed = 1;
		// 1) data loader 
		Instances data=new Instances(new BufferedReader(new FileReader("D:\\Weka-3-9\\data\\weather.numeric.arff")));
		if(isRemove){
			Remove filter = new Remove();
			filter.setAttributeIndices("1");
			filter.setInputFormat(data);
			data = Filter.useFilter(data, filter);		
		}	
		int trainSize = (int)Math.round(data.numInstances() * 66 / 100);
		int testSize = data.numInstances() - trainSize;
		data.randomize(new java.util.Random(seed));		
		Instances train = new Instances (data, 0 ,trainSize);
		Instances test  = new Instances (data, trainSize ,testSize);

		// 2) class assigner
		train.setClassIndex(train.numAttributes()-1);
		test.setClassIndex(test.numAttributes()-1);
		
		// 3) cross validate setting  
		Evaluation eval=new Evaluation(train);
		OneR model=new OneR();
		/************************
		 * MinBucketSize ���� 
		 ************************/
		model.setMinBucketSize(minBucketSize);

		// 4) model run 
		model.buildClassifier(train);
		
		// 5) evaluate
		eval.evaluateModel(model, test);
		
		// 6) print Result text
		System.out.println("\t�з���� ������ �� �� : " + (int)eval.numInstances() + 
				           ", ���з� �Ǽ� : " + (int)eval.correct() + 
				           ", ���з��� : " + String.format("%.1f",eval.correct() / eval.numInstances() * 100) +" %"+ 
				           ", minBucketSize : " + minBucketSize + 
				           ", �з��� : OneR" + ", Ȧ��ƿ� ����"); 	
		System.out.println(model);
	}
	
	public void diabeteCrossValidationOneR(Classifier obj, boolean isUseTrainingSet, int minBucketSize) throws Exception{
		int seed = 1;
		int numfolds = 10;
		int numfold = 0;
		
		// 1) data loader 
		Instances data=new Instances(new BufferedReader(new FileReader("D:\\Weka-3-9\\data\\diabetes.arff")));
		Instances train = null;
		Instances test  = null;
		if(isUseTrainingSet){ 
			// �м���� �����͸� �״�� �Ʒ�/�׽�Ʈ �����ͷ� ���� (Use training set)
			train = new Instances(data);
			test  = new Instances(data);			
		}else{ 
			// crossValidation
			train = data.trainCV(numfolds, numfold, new Random(seed));
			test  = data.testCV (numfolds, numfold);
		}
		
		// 2) class assigner
		train.setClassIndex(train.numAttributes()-1);
		test.setClassIndex(test.numAttributes()-1);
		
		// 3) cross validate setting  
		Evaluation eval=new Evaluation(train);
		Classifier model = obj;
		if(obj instanceof OneR){
			/************************
			 * MinBucketSize ���� ����
			 ************************/
			((OneR)model).setMinBucketSize(minBucketSize);
		}		
		if(!isUseTrainingSet) // Use Training set Only (�Ʒõ����� ��) �ƴ� ��츸 ����
			eval.crossValidateModel(model, train, numfolds, new Random(seed)); 

		// 4) model run 
		model.buildClassifier(train);
		
		// 5) evaluate
		eval.evaluateModel(model, test);
		
		// 6) print Result text
		System.out.println("\t�з���� ������ �� �� : " + (int)eval.numInstances() + 
		           ", ���з� �Ǽ� : " + (int)eval.correct() + 
		           ", ���з��� : " + String.format("%.1f",eval.correct() / eval.numInstances() * 100) +" %"+ 
		           ", minBucketSize : " + minBucketSize + 
		           ", �з��� : " + ((obj instanceof ZeroR)?"ZeroR":"OneR") +
		           ", " + ((isUseTrainingSet)?"Use Training set Only (�Ʒõ����� ��)":"crossvalidation (��������)") + " ����"  
		           );
		System.out.println(model);
	}	
	
}
