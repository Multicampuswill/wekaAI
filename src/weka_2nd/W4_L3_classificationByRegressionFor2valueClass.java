package weka_2nd;

import java.io.*;
import java.util.Random;
import weka.classifiers.*;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.rules.OneR;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AddClassification;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;

public class W4_L3_classificationByRegressionFor2valueClass {

	public static void main(String args[]) throws Exception{
		W4_L3_classificationByRegressionFor2valueClass obj = new W4_L3_classificationByRegressionFor2valueClass();
		String fileName= "diabetes";
		System.out.println(fileName + " : ");
		
		/*****************************************************************************
		 *  LinearRegression ������ ����
		 *  https://svn.cms.waikato.ac.nz/svn/weka/branches/stable-3-8/weka/lib/ �����Ͽ�
		 *  arpack_combined.jar, mtj.jar, core.jar �� �ܺ� jar �� ����Ʈ �ؾ� �Ѵ�.
		 ******************************************************************************/
		// 1) �̻������� �������� ������ �����ͼ� ��ȯ �� ȸ�ͽ� ���� 
		Instances filtterdData = obj.diabeteRegressionForNominalToBinaryFilter(fileName,new LinearRegression());

		// 2) AddClassification,  NumericToNominal, Remove ���� ����
		filtterdData = obj.applyFilters(new LinearRegression(), filtterdData, "9", "1-8");
		
		// 3) OneR �з��� ���� �Ӱ��� ����
//		obj.diabeteOneRForAddclassificationFilter(new OneR(),filtterdData,6);  // minBucketSize ������ ������ ����
		obj.diabeteOneRForAddclassificationFilter(new OneR(),filtterdData,100);// ������ �غ� ���� minBucketSize Ȯ��  
	}

	public Instances diabeteRegressionForNominalToBinaryFilter(String fileName, Classifier model) throws Exception{
		int seed = 1;
		int numfolds = 10;
		int numfold = 0;		
		// 1) data loader 
		Instances data=new Instances(new BufferedReader(new FileReader("D:\\Weka-3-9\\data\\"+fileName+".arff")));
		/*****************************
		 * NominalToBinary ���� ���� ����
		 *****************************/
		NominalToBinary filter = new NominalToBinary(); // unsupervised �� ���� (supervised ���� ������ ���� ����
		filter.setAttributeIndices("last");
		filter.setInputFormat(data);
		data = Filter.useFilter(data, filter);
		/*****************************
		 * NominalToBinary ���� ���� ����
		 *****************************/

		Instances train = data.trainCV(numfolds, numfold, new Random(seed));
		Instances test  = data.testCV (numfolds, numfold);
				
		// 2) class assigner
		train.setClassIndex(train.numAttributes()-1);
		test. setClassIndex(test. numAttributes()-1);
		
		// 3) cross validate setting  
		Evaluation eval=new Evaluation(train);
		eval.crossValidateModel(model, train, numfolds, new Random(seed));
//		Classifier model=classifier; // �Ű��������� ���� ������ model��ü�� ���� ���		
		
		// 4) model run 
		model.buildClassifier(train);
		
		// 5) evaluate
		eval.evaluateModel(model, test);
		
		// 6) print Result text
		System.out.println("\n**********************************************************************");
		System.out.println("\n         1) �̻������� �������� ������ �����ͼ� ��ȯ �� ȸ�ͽ� ����");
		System.out.println("\n**********************************************************************");
		System.out.println("1-1) NominalToBinary ������ data �Ӽ����� : " + data.numAttributes());
		System.out.println("1-2) ȸ�ͽ� model : " + model.toString() +"\n"+eval.toSummaryString()); 

		// 7) NominalToBinary ����� instances (�����ͼ�Ʈ) ��ȯ
		return data;
	}

	public void diabeteOneRForAddclassificationFilter(Classifier model, Instances filtterdData, int minBuckeSize) throws Exception{
		int seed = 1;
		int numfolds = 10;
		int numfold = 0;		
		// 1) data loader 
		Instances data=filtterdData;
		data.setClassIndex(0);

		Instances train = data.trainCV(numfolds, numfold, new Random(seed));
		Instances test  = data.testCV (numfolds, numfold);
		
		// 2) class assigner (���͸� �������� class �Ӽ��� 1��°(index=0)�� �Ű�����)
		train.setClassIndex(0);
		test. setClassIndex(0);
		
		// 3) cross validate setting  
		Evaluation eval=new Evaluation(train);
		OneR classifier=(OneR)model;				
		/**********************************************************
		 * ������ ������ ���� minBuckeSize �� (�Ű������� ����) 100  ���� ���� ����
		 *********************************************************/
		classifier.setMinBucketSize(minBuckeSize);	
		/**********************************************************
		 * ������ ������ ���� minBuckeSize �� (�Ű������� ����) 100  ���� ���� ����
		 *********************************************************/
		
		// 4) model run 
		classifier.buildClassifier(train);
		
		// 5) evaluate
		eval.evaluateModel(classifier, test);
		
		// 6) print Result text
		System.out.println("\n**********************************************************************");
		System.out.println("\n          3) OneR �з��� ���� �Ӱ��� ����");
		System.out.println("\n**********************************************************************");
		System.out.println("3) minBuckeSize : "+ minBuckeSize + "\n classifier : " + classifier.toString() +"\n"+eval.toSummaryString()); // ȸ�ͺм��� ���з��� ���� �������� ���� ȸ�͹����� ������ ��ǥ�� �߿�
	}
	
	/************************
	 * ���� ���͸� �����ϱ� ���� �޼ҵ�
	 ************************/
	public Instances applyFilters(Classifier model, Instances data, String transIndicice, String removeIndices) throws Exception{
		System.out.println("\n**********************************************************************");
		System.out.println("\n          2) AddClassification,  NumericToNominal, Remove ���� ����");
		System.out.println("\n**********************************************************************");
		data.setClassIndex(data.numAttributes()-1);
		// 1) AddClassification (���ڷθ� �� ������� classification �Ӽ� �߰�)
		AddClassification addfilter = new AddClassification();
		addfilter.setClassifier(model);
		addfilter.setOutputClassification(true);
		addfilter.setInputFormat(data);
		data = Filter.useFilter(data, addfilter);
		System.out.println("2-1) AddClassification ������ data �Ӽ����� : " + data.numAttributes());
		
		// 2) NumericToNominal (�̻������� �и��� ���������� ��������� ��ȯ)
		NumericToNominal changeTypefilter = new NumericToNominal(); 
		changeTypefilter.setAttributeIndices(transIndicice);
		changeTypefilter.setInputFormat(data);
		data = Filter.useFilter(data, changeTypefilter);
		System.out.println("2-2) NumericToNominal ������ data �Ӽ����� : " + data.numAttributes());
		
		// 3) Remove (���������� classification �� ����� ���Ӽ� ����)
		Remove filter = new Remove(); 
		filter.setAttributeIndices(removeIndices);
		filter.setInputFormat(data);
		data = Filter.useFilter(data, filter);
		System.out.println("2-3) Remove ������ data �Ӽ����� : " + data.numAttributes());
		
		return data;
	}

}
