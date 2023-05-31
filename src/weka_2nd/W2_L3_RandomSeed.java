package weka_2nd;

import java.io.*;
import java.util.Random;

import org.apache.commons.math3.stat.descriptive.AggregateSummaryStatistics;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SimpleLogistic;
import weka.classifiers.trees.J48;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.*;
import weka.core.*;

public class W2_L3_RandomSeed {
	
	double correctRatio = 0.0;
	public static void main(String args[]) throws Exception{
		W2_L3_RandomSeed obj = new W2_L3_RandomSeed();
		double sum[] = new double[10];
		/**********************************************************
		 * RandomSeed�� 1�� �������� ��Ȯ���� ����� ������� ���� ����
		 **********************************************************/
		for(int x=1 ; x<=10 ; x++){
			System.out.print("90% split, RandomSeed = " + x);
			sum[x-1] = obj.randomSeed(90,x);
		}	
		obj.aggregateValue(sum);
		/**********************************************************
		 * RandomSeed�� 1�� �������� ��Ȯ���� ����� ������� ���� ����
		 **********************************************************/
	}
	
	public double randomSeed(int percent, int seed) throws Exception{
		// 1) data loader 
		Instances data=new Instances(new BufferedReader(new FileReader("D:\\Weka-3-9\\data\\segment-challenge.arff")));

		int trainSize = (int)Math.round(data.numInstances() * percent / 100);
		int testSize = data.numInstances() - trainSize;
		data.randomize(new java.util.Random(seed));
		
		Instances train = new Instances (data, 0 ,trainSize);
		Instances test  = new Instances (data, trainSize ,testSize);
		
		// 2) class assigner
		train.setClassIndex(train.numAttributes()-1);
		test.setClassIndex(test.numAttributes()-1);
		
		// 3) learn and evaluate setting  
		Evaluation eval=new Evaluation(train);
		Classifier model=new J48();
		
		// 4) model run 
		model.buildClassifier(train);
		
		// 5) evaluate
		eval.evaluateModel(model, test);
		
		// 6) print Result text
		this.correctRatio += eval.correct() / eval.numInstances() * 100; // �з���Ȯ�� ����
		System.out.println("\t�з���� �׽�Ʈ ������ �� �� : " + eval.numInstances() + ", ���з� �Ǽ� : " + eval.correct() + ", �з���Ȯ�� : " + eval.pctCorrect() +" %"); 	
	
		return eval.pctCorrect(); // ���з��� ��ȯ
	}
	
	/**
	 * common-math jar �ٿ�ε� ��ġ : http://apache.mirror.cdnetworks.com/commons/math/binaries/
	 * **/
	public void aggregateValue(double[] sum){
		AggregateSummaryStatistics aggregate = new AggregateSummaryStatistics();
		SummaryStatistics sumObj = aggregate.createContributingStatistics();
		for(int i = 0; i < sum.length; i++)  sumObj.addValue(sum[i]); 

		System.out.println("��� : " + String.format("%.1f",aggregate.getMean()) + " %, �л� : " + String.format("%.1f",aggregate.getStandardDeviation())  + " %");
	}
}
