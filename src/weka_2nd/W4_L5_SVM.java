package weka_2nd;

import java.io.*;
import java.util.*;


import weka.classifiers.*;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.*;

public class W4_L5_SVM {
	 
	 public static void main(String args[]) throws Exception{

		W4_L5_SVM obj = new W4_L5_SVM();
		System.out.println("=============================================================================");
		System.out.println("\t 1) 1�� �����ͼ�Ʈ�� 4�� �з��⸦ �迭�� ������ �� ���� ȣ��");
		System.out.println("=============================================================================");
	    String fileNames[] = {"credit-g"}; // ����� ����ϴ� 1�� arff ������ �迭�� ����
	    Classifier[] models = {new IBk(),new J48(), new Logistic(), new SMO()}; // ���� ���׺� ���� 4�� �з��⸦ �迭�� ����
//		String fileNames[] = {"diabetes"}; Classifier[] models = {new Logistic()};

	    for(String fileName : fileNames){
			for(Classifier model : models){
				System.out.println(fileName + " : ");
				double crossValidation = obj.crossValidataion(fileName,model);
				double useTrainingSet  = obj.useTrainingSet(fileName, model);
				
				obj.printOverFitting(model, crossValidation, useTrainingSet);
			}// end-for-models	
		}// end-for-fileNames	 
	}

	/*****************************
	 * ��������
	 *****************************/
	public double crossValidataion(String fileName, Classifier model) throws Exception{
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
		eval.crossValidateModel(model, train, numfolds, new Random(seed));	  

		// 4) model run 
		model.buildClassifier(train);
		   
		// 5) evaluate
		eval.evaluateModel(model, test);
		
		// 6) print Result text (�з��� ���з��� �� ����������� ���)
		return this.printClassfiedInfo(model, eval);
			   			   
	}

	/*****************************
	 * GUI use TrainingSet
	 *****************************/
	public double useTrainingSet(String fileName, Classifier model) throws Exception{
		
		// 1) data loader 
		Instances data=new Instances(new BufferedReader(new FileReader("D:\\Weka-3-9\\data\\"+fileName+".arff")));
		
		// 2) class assigner
		data.setClassIndex(data.numAttributes()-1);
		
		// 3) ������ü ����
		Evaluation eval = new Evaluation(data);

		// 4) model run 
		model.buildClassifier(data);
		   
		// 5) evaluate
		eval.evaluateModel(model, data);
		
		// 6) print Result text (�з��� ���з��� �� ����������� ���)
		return this.printClassfiedInfo(model, eval);
	}
	
	/*****************************
	 * 6) �з��� ���з��� �� ����������� ���
	 *****************************/
	public double printClassfiedInfo(Classifier model, Evaluation eval){
		System.out.print("Correctly Classified Instances : " + String.format("%.2f",eval.pctCorrect()));
		System.out.print(", Root mean squared error  :" + String.format("%.2f",eval.rootMeanSquaredError()));
		System.out.println(", (" + getModelName(model) + ")");
		
		return eval.pctCorrect();
	}

	/**************************************
	 * 7) Print OverFitting 
	**************************************/
	public void printOverFitting(Classifier model, double crossValidation, double useTrainingSet) throws Exception{		
		System.out.println(" ==> Overfitting (difference) :" + 
	                       String.format("%.2f",useTrainingSet - crossValidation) + " % (" + 
				           getModelName(model)+ ")");
		System.out.println("");
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
		else if ( model instanceof  IBk)
			modelName = "IBk";
		else if ( model instanceof  SMO)
			modelName = "SMO";
		return modelName;
	}
}
