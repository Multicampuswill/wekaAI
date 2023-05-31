package weka_2nd;

import java.io.*;
import java.util.*;

import javax.swing.plaf.synth.SynthSeparatorUI;

import weka.classifiers.*;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.evaluation.output.prediction.PlainText;
import weka.classifiers.functions.Logistic;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.J48;
import weka.core.*;

public class WekaCommon {
	 String[] labels = null;
	 
	 public static void main(String args[]) throws Exception{
		/*****************************************************************************
		 *  LinearRegression ������ ����
		 *  https://svn.cms.waikato.ac.nz/svn/weka/branches/stable-3-8/weka/lib/ �����Ͽ�
		 *  arpack_combined.jar, mtj.jar, core.jar �� �ܺ� jar �� ����Ʈ �ؾ� �Ѵ�.
		 ******************************************************************************/
		 WekaCommon obj = new WekaCommon();
//	     String fileNames[] = {"glass","labor","breast-cancer"}; // ����� ����ϴ� 3�� arff ������ �迭�� ����
	     String fileNames[] = {"breast-cancer"}; // ����� ����ϴ� 3�� arff ������ �迭�� ����
	     obj.logisticRegressions(fileNames);
	 }

	 /**
	  * �з��� �迭ȭ ���� : https://www.programcreek.com/2013/01/a-simple-machine-learning-example-in-java/
	  * ���� ���� (�׳� ����) : https://www.programcreek.com/java-api-examples/?api=weka.classifiers.Evaluation
	  * **/
	 public void logisticRegressions(String[] fileNames) throws Exception{
//	    Classifier[] models = {new Logistic(),new J48(), new ZeroR(), new NaiveBayes()}; // ���� ���׺� ���� �з��⸦ �迭�� ����
	    Classifier[] models = {new Logistic()}; // ���� ���׺� ���� �з��⸦ �迭�� ����
		  
		for(String fileName : fileNames){
			System.out.println(fileName + " : ");
			this.logisticRegression(fileName,models);    
		}
	 }
			 
	public void logisticRegression(String fileName, Classifier[] models) throws Exception{
		int numfolds = 10;
		int numfold = 0;
		int seed = 1;
		  
		// 1) data loader 
		Instances data=new Instances(new BufferedReader(new FileReader("D:\\Weka-3-9\\data\\"+fileName+".arff")));
		data.setClassIndex(data.numAttributes()-1); 
		
		Instances train = data.trainCV(numfolds, numfold, new Random(seed));
		Instances test  = data.testCV (numfolds, numfold);
		
		// 2) class assigner
		train.setClassIndex(train.numAttributes()-1);
		test.setClassIndex(test.numAttributes()-1);
		  
		// 3) cross validate setting  
		Evaluation eval=new Evaluation(train);
		  
		// �з��⺰ ���� ����
		for(Classifier model : models){ // models �迭�� �з��⸦ model �̶� ��ü�� �ϳ��� �����Ͽ� ���� (index ���� ���� ��)
			/************************
			* plainText ��ü��� ���� ���� : https://stackoverflow.com/questions/21424248/get-risk-predictions-in-weka-using-own-java-code
			************************/
			StringBuffer predictionSB = new StringBuffer();
			Range attributesToShow = null;
			Boolean outputDistributions = new Boolean(true);
			
			PlainText predictionOutput = new PlainText();
			predictionOutput.setBuffer(predictionSB);
			predictionOutput.setOutputDistribution(true);
			/************************
			* plainText ��ü��� ���� ����
			************************/
			   
			eval.crossValidateModel(model, train, numfolds, new Random(seed), 
					                predictionOutput, attributesToShow, outputDistributions);
     		/************************
			* plainText ��ü��� ����
			************************/
			System.out.println(predictionSB);
			/************************
			* plainText ��ü��� ����
			************************/
	
			// 4) model run 
			model.buildClassifier(train);
			   
			// 5) evaluate
			eval.evaluateModel(model, test);
			
			
			// 6) print Result text
			System.out.print("Correctly Classified Instances : " + String.format("%.2f",eval.pctCorrect()));
			System.out.print(", Root mean squared error  :" + String.format("%.2f",eval.rootMeanSquaredError()));
			System.out.println(", (" + getModelName(model) + ")");

			   
			// 7) set Labels
			this.setLabels(data);
			   
			// 8) print out (with test)
//			this.printTestInstances(test, eval, model);
			
			// 8) print out (with prediction : Prediction ��ä�δ� distribution �� ã�� �� ����. (���� ��ü�ȿ� ���� ������ ����޼ҵ� ����)
			// this.printPrediction(eval);
		} // end-of-for-model
	}
	
	/*****************************
	 * Model Name
	 *****************************/
	public String getModelName(Classifier model){
		String modelName = "";
		if ( model instanceof  Logistic)
			modelName = "Logistic";
		else if ( model instanceof  J48)
			modelName = "J48";
		else if ( model instanceof  ZeroR)
			modelName = "ZeroR";
		else if ( model instanceof  NaiveBayes)
			modelName = "NaiveBayes";
	return modelName;
	}

	 /*****************************
	  * Labels = Indices setting
	  *****************************/
	public void setLabels(Instances data){
		int labelSize = data.classAttribute().numValues();
		this.labels = new String[labelSize];
		for(int x=0 ; x < labelSize ; x++){
			labels[x] = data.classAttribute().value(x);
		}
	}

	 /*****************************
	  * Print by Test data
	  *****************************/
	 public void printTestInstances(Instances test, Evaluation eval, Classifier model) throws Exception{
		 for (int x=0; x<test.size() ; x++){
			 Instance oneData = test.instance(x);
			 int actual    = (int)oneData.classValue();
			   
			 Prediction prediction = eval.predictions().get(x);
			 int predicted = (int)prediction.predicted(); 
			   
			 double[] distribution = model.distributionForInstance(oneData);
			 System.out.print((x+1) + " ");
			 System.out.print( 
						 (actual+1)    + ":" + labels[actual] + " " +
						 (predicted+1) + ":" + labels[predicted] + " " +
						 ((actual == predicted)?" ":"+") + " " + 
						 String.format("%.2f",distribution[0]) + " " +
						 String.format("%.2f",distribution[1])
			 );  
			 System.out.println("");
		 }    
	 }

	 /*****************************
	  * Print by Prediction object
	  *****************************/
	 public void printPrediction(Evaluation eval){
		 ArrayList<Prediction> list = eval.predictions();
		 int x=0;
		 for (Prediction prediction : list) {
			 x++;
			 int actual = (int)prediction.actual();
			 int predicted = (int)prediction.predicted();
			 System.out.print((x+1) + " ");
			 System.out.print( 
						 (prediction.actual()+1)    + ":" + labels[actual] + " " +
						 (prediction.predicted()+1) + ":" + labels[predicted] + " " +
						 ((actual == predicted)?"":"+") + " "  
						 ); 
			 System.out.println("");
		 }
	 }
	 
	 /*****************************
	  * Logistic coefficients info
	  *****************************/
	 public HashMap<String, Double> fetchCoefficientsInfo(Classifier model, Instances data){
		HashMap<String, Double> coeffMap = new HashMap<String, Double>();
		double[][] coeff = ((Logistic)model).coefficients();
		Enumeration<Attribute> enums = data.enumerateAttributes();
		while (enums.hasMoreElements()) {
			Attribute attribute = (Attribute) enums.nextElement();			
			int col = attribute.index()+1;
			System.out.println(attribute.name() + " : " + String.format("%.4f",coeff[col][0]));
			coeffMap.put(attribute.name(), Double.valueOf(coeff[col][0]));
		}
		System.out.println("Intercept : " + String.format("%.4f",coeff[0][0])); // intercept = bias
		coeffMap.put("Intercept", Double.valueOf(coeff[0][0]));
		System.out.println(model);		   
		return coeffMap;
	 }

}
