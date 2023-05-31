package weka_2nd;

import java.io.*;
import java.util.*;


import weka.classifiers.*;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.functions.Logistic;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.J48;
import weka.core.*;

public class W4_L4_logisticRegression {
	 String[] labels = null;
	 
	 public static void main(String args[]) throws Exception{
		/*****************************************************************************
		 *  LinearRegression ������ ����
		 *  https://svn.cms.waikato.ac.nz/svn/weka/branches/stable-3-8/weka/lib/ �����Ͽ�
		 *  arpack_combined.jar, mtj.jar, core.jar �� �ܺ� jar �� ����Ʈ �ؾ� �Ѵ�.
		 ******************************************************************************/
		 W4_L4_logisticRegression obj = new W4_L4_logisticRegression();
	     obj.logisticRegressions();
	 }

	 /**
	  * �з��� �迭ȭ ���� : https://www.programcreek.com/2013/01/a-simple-machine-learning-example-in-java/
	  * ���� ���� (�׳� ����) : https://www.programcreek.com/java-api-examples/?api=weka.classifiers.Evaluation
	  * **/
	 public void logisticRegressions() throws Exception{
		System.out.println("=============================================================================");
		System.out.println("\t 1) 3�� �����ͼ�Ʈ�� 4�� �з��⸦ �迭�� ������ �� ���� ȣ��");
		System.out.println("=============================================================================");
	    String fileNames[] = {"glass","labor","breast-cancer"}; // ����� ����ϴ� 3�� arff ������ �迭�� ����
	    Classifier[] models = {new Logistic(),new J48(), new ZeroR(), new NaiveBayes()}; // ���� ���׺� ���� 4�� �з��⸦ �迭�� ����
//		String fileNames[] = {"diabetes"}; Classifier[] models = {new Logistic()};
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
			// 3) �������� ����
			eval.crossValidateModel(model, train, numfolds, new Random(seed));
	
			// 4) model run 
			model.buildClassifier(train);
			   
			// 5) evaluate
			eval.evaluateModel(model, test);
			
			// 6) print Result text (�з��� ���з��� �� ����������� ���)
			this.printClassfiedInfo(model, eval);
			   			   
			// 7) print out (with test)
//			this.printDistribution(test, eval, model);
			
			// 8) ������ƽ ȸ�ͽ��� ���κ����� ������ ����
//			if( model instanceof Logistic)
//				this.fetchCoefficientsInfo(model, data);
		} // end-of-for-model
	}

	
	/*****************************
	 * 2) �з��� ���з��� �� ����������� ���
	 *****************************/
	public void printClassfiedInfo(Classifier model, Evaluation eval){
		System.out.println("=============================================================================");
		System.out.println("\t 2) �з��� ���з��� �� ����������� ���");
		System.out.println("=============================================================================");
		System.out.print("Correctly Classified Instances : " + String.format("%.2f",eval.pctCorrect()) + " %");
		System.out.print(", Root mean squared error  :" + String.format("%.2f",eval.rootMeanSquaredError()));
		System.out.println(", (" + getModelName(model) + ")");
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
	
	 /**************************************
	  * 3) Print distribution by Test data
	  **************************************/
	 public void printDistribution(Instances test, Evaluation eval, Classifier model) throws Exception{
		 System.out.println("=============================================================================");
		 System.out.println("\t 3) distribution ���");
		 System.out.println("=============================================================================");
		 
		 this.setLabels(test); // lable ���� (�ű��߰� �޼ҵ�)
		 for (int x=0; x<test.size() ; x++){
			 Instance oneData = test.instance(x);
			 int actual    = (int)oneData.classValue();  // ��ǥ���� (class) �������� actual �� �Ҵ�
			   
			 Prediction prediction = eval.predictions().get(x); // �з��⿡�� ����� ������� prediction �� �Ҵ�
			 int predicted = (int)prediction.predicted(); 
			   
			 double[] distribution = model.distributionForInstance(oneData); // �𵨿��� ����� distribution �� distributio �� �Ҵ�
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
	  * Prediction ��ä�δ� distribution �� ã�� �� ����. (���� ��ü�ȿ� ���� ������ ����޼ҵ� ����)
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
	  * 4) Logistic coefficients info
	  *****************************/
	 public HashMap<String, Double> fetchCoefficientsInfo(Classifier model, Instances data){
		System.out.println("=============================================================================");
		System.out.println("\t 4) Logistic coefficients info ���");
		System.out.println("=============================================================================");
		HashMap<String, Double> coeffMap = new HashMap<String, Double>();
		double[][] coeff = ((Logistic)model).coefficients();       // ������ƽ ȸ�ͺз��⿡ ����� ������ ������ �Ҵ�
		Enumeration<Attribute> enums = data.enumerateAttributes(); // ������  �Ҵ�
		while (enums.hasMoreElements()) {
			Attribute attribute = (Attribute) enums.nextElement(); // ������ ����			
			int col = attribute.index()+1;
			System.out.println(attribute.name() + " : " + String.format("%.4f",coeff[col][0])); // ����� ������ �� ������ ���
			coeffMap.put(attribute.name(), Double.valueOf(coeff[col][0])); // ����� ������ �� ������ ����
		}
		System.out.println("Intercept : " + String.format("%.4f",coeff[0][0])); // intercept = bias
		coeffMap.put("Intercept", Double.valueOf(coeff[0][0])); // �Ǹ����� intercept �� ���� ������ ����
		System.out.println(model);		   
		return coeffMap;
	 }

}
