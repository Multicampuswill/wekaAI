package weka_2nd;

import java.io.*;
import java.util.*;


import weka.classifiers.*;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.core.*;

public class W4_L6_Ensemble {
	 
	 public static void main(String args[]) throws Exception{

		W4_L6_Ensemble obj = new W4_L6_Ensemble();
		System.out.println("=============================================================================");
		System.out.println("\t 1) AdaboostM1 �� �з��� ���� ���з� ��");
		System.out.println("=============================================================================");
	    String fileName = "diabetes"; // ����� ����ϴ� 1�� arff ������ �迭�� ����
	    Classifier[] models = {new DecisionStump(), new ZeroR(), new IBk(), new J48(), new Logistic(), new SMO()}; // AdaBoostM1 �� �з��� 4���� �迭�� ����

		for(Classifier model : models){
			System.out.println(fileName + " : ");
			obj.boost_Ensemble_crossvalidation(fileName,model,1,10);
		}// end-for-models	

		System.out.println("=============================================================================");
		System.out.println("\t 1) AdaboostM1 �� ������ ����");
		System.out.println("=============================================================================");
		fileName = "credit-g"; 
		for(Classifier model : models){
			System.out.println(fileName + " : ");
			double crossValidation = obj.boost_Ensemble_crossvalidation(fileName,model,10,10);
			double useTrainingSet  = obj.useTrainingSet(fileName, model);
			
			obj.printOverFitting(model, crossValidation, useTrainingSet);
		}// end-for-models	
	}

	/*****************************
	 * boost Ensemble
	 *****************************/
	public double boost_Ensemble_crossvalidation(String fileName, Classifier model, int start, int maxLoop) throws Exception{
		int numfolds = 10;
		int numfold = 0;
		int seed = 1;
		  
		// 1) data loader 
		Instances data=new Instances(new BufferedReader(new FileReader("D:\\Weka-3-9\\data\\"+fileName+".arff")));
//		data.setClassIndex(data.numAttributes()-1); 
		
		Instances train = data.trainCV(numfolds, numfold, new Random(seed));
		Instances test  = data.testCV (numfolds, numfold);
		
		// 2) class assigner
		train.setClassIndex(train.numAttributes()-1);
		test.setClassIndex(test.numAttributes()-1);
		
		double rslt = 0;

		for (int i = start; i <= maxLoop; i++) 
		{
			// 3) cross validate setting  
			Evaluation eval=new Evaluation(train);		  
	
			// 4) model run 
			AdaBoostM1 boost = new AdaBoostM1();
			boost.setClassifier(model);
			eval.crossValidateModel(boost, train, numfolds, new Random(seed));
		
			boost.setNumIterations(i);
			boost.buildClassifier(train);

			// 5) evaluate
			eval.evaluateModel(boost, test);
			
			// 6) print Result text (�з��� ���з��� ���)
			rslt = this.printClassfiedInfo(boost, eval);
		}  	
		return rslt;
	}

	/*****************************
	 * GUI use TrainingSet
	 *****************************/
	public double useTrainingSet(String fileName, Classifier model) throws Exception{
		Instances data=new Instances(new BufferedReader(new FileReader("D:\\Weka-3-9\\data\\"+fileName+".arff")));
		
		// 2) class assigner
		data.setClassIndex(data.numAttributes()-1);
		
		// 3) ������ü ����
		Evaluation eval = new Evaluation(data);

		// 4) model run 
		AdaBoostM1 boost = new AdaBoostM1();
		boost.setClassifier(model);
		boost.buildClassifier(data);
		   
		// 5) evaluate
		eval.evaluateModel(boost, data);
		
		// 6) print Result text (�з��� ���з��� �� ����������� ���)
		return this.printClassfiedInfo(boost, eval);
	}
	
	/*****************************
	 * 6) �з��� ���з��� �� numIteration ���
	 *****************************/
	public double printClassfiedInfo(AdaBoostM1 boost, Evaluation eval){
		System.out.print("Correctly Classified Instances : " + String.format("%.2f",eval.pctCorrect()));
		System.out.print(", numIteration  :" + boost.getNumIterations());
		System.out.println(", (" + getModelName(boost.getClassifier()) + ")");
		
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
		else if ( model instanceof  DecisionStump)
			modelName = "DecisionStump";
		else if ( model instanceof  ZeroR)
			modelName = "ZeroR";
		return modelName;
	}
}
