package weka_2nd;

import java.io.*;
import java.util.Random;
import weka.classifiers.*;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.trees.M5P;
import weka.core.*;

public class W4_L2_LinearRegression_M5P {

	public static void main(String args[]) throws Exception{
		W4_L2_LinearRegression_M5P obj = new W4_L2_LinearRegression_M5P();
		String fileName= "cpu";
		System.out.println(fileName + " : ");
		
		/*****************************************************************************
		 *  LinearRegression ������ ����
		 *  https://svn.cms.waikato.ac.nz/svn/weka/branches/stable-3-8/weka/lib/ �����Ͽ�
		 *  arpack_combined.jar, mtj.jar, core.jar �� �ܺ� jar �� ����Ʈ �ؾ� �Ѵ�.
		 ******************************************************************************/
		obj.cpuRegression(fileName,new LinearRegression());  
		
		// M5P�� ���� 3�� jar �� ��� ���డ����.
		obj.cpuRegression(fileName,new M5P());  
	}

	public void cpuRegression(String fileName, Classifier model) throws Exception{
		int seed = 1;
		int numfolds = 10;
		int numfold = 0;		
		// 1) data loader 
		Instances data=new Instances(new BufferedReader(new FileReader("D:\\Weka-3-9\\data\\"+fileName+".arff")));

		Instances train = data.trainCV(numfolds, numfold, new Random(seed));
		Instances test  = data.testCV (numfolds, numfold);
		
		// 2) class assigner
		train.setClassIndex(train.numAttributes()-1);
		test. setClassIndex(test. numAttributes()-1);
		
		// 3) cross validate setting  
		Evaluation eval=new Evaluation(train);
//		Classifier model=classifier; // �Ű��������� ���� ������ model��ü�� ���� ���		
		
		// 4) model run 
		model.buildClassifier(train);
		
		// 5) evaluate
		eval.evaluateModel(model, test);
		
		// 6) print Result text
		System.out.println("model : " + model.toString() +"\n"+eval.toSummaryString()); // ȸ�ͺм��� ���з��� ���� �������� ���� ȸ�͹����� ������ ��ǥ�� �߿�

		// 7) �н��� �𵨷� �߼� ����
		this.trend(model, data);
	}

	public void trend(Classifier model, Instances data) throws Exception{
		double differ = 0.0;
		double sumDifferABS = 0.0;
		double classValue = 0.0;
		double result = 0.0;
		for(int x=0 ; x < data.size() ; x++){
			Instance row = data.get(x);
			/**************************
			 * ������ �𵨷� class ��� ����
			 **************************/
			result = model.classifyInstance(row);
			/**************************
			 * ������ �𵨷� class ��� ����
			 **************************/
			classValue = row.valueSparse(row.numAttributes()-1);
			differ = result - classValue;
			sumDifferABS += Math.abs(differ); // ������ ���밪 ����
			System.out.println( (x+1) + " : " + String.format("%.1f",classValue) + 
					                   " => " + String.format("%.1f",result) + 
					                   " �Ǻ� ���� :" + String.format("%.1f",differ) );		
		}			
		System.out.println( "������� : " + sumDifferABS/data.size());		
		
		if( model instanceof M5P){
			M5P m5p = (M5P)model; 
			// ������ȸ��..
		}else{
			LinearRegression linear = (LinearRegression)model;
			// ������ȸ��..
		}		
		
	}
}
