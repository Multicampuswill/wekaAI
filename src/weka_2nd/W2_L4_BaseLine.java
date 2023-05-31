package weka_2nd;

import java.io.*;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.rules.PART;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.J48;
import weka.core.*;

public class W2_L4_BaseLine {
	
	public W2_L4_BaseLine(){
		try{
			Classifier model = new J48();
			model.buildClassifier(null);
		}catch(Exception e){
			System.out.println("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
		}
	}
	
	public static void main(String args[]) throws Exception{
		W2_L4_BaseLine obj = new W2_L4_BaseLine();		
		/**********************************************************
		 * ���غз��� ZeroR �� �ʵη� 4�� �з��� ��Ȯ�� ��� ����
		 **********************************************************/
		System.out.print("ZeroR");obj.baseLine(new ZeroR(),66);
		System.out.print("J48");obj.baseLine(new J48(),66);
		System.out.print("NaiveBayes");obj.baseLine(new NaiveBayes(),66);
		System.out.print("IBk");obj.baseLine(new IBk(),66);
		System.out.print("PART");obj.baseLine(new PART(),66);
		/**********************************************************
		 * ���غз��� ZeroR �� �ʵη� 4�� �з��� ��Ȯ�� ��� ����
		 **********************************************************/
	}
	
	public void baseLine(Classifier model, int percent) throws Exception{
		// 1) data loader 
		Instances data=new Instances(new BufferedReader(new FileReader("D:\\Weka-3-9\\data\\supermarket.arff")));

		int trainSize = (int)Math.round(data.numInstances() * percent / 100);
		int testSize = data.numInstances() - trainSize;
		data.randomize(new java.util.Random(1));
		
		Instances train = new Instances (data, 0 ,trainSize);
		Instances test  = new Instances (data, trainSize ,testSize);
		
		// 2) class assigner
		train.setClassIndex(train.numAttributes()-1);
		test.setClassIndex(test.numAttributes()-1);
		
		// 3) object creation  
		Evaluation eval=new Evaluation(train);
		
		// 4) model run 
		model.buildClassifier(train);
		
		// 5) evaluate
		eval.evaluateModel(model, test);
		
		// 6) print Result text
		System.out.println("\t�з���� ������ �� �� : " + eval.numInstances() + ", ���з� �Ǽ� : " + eval.correct() + ", �з���Ȯ�� : " + eval.correct() / eval.numInstances() * 100 +" %"); 	
	}
}
