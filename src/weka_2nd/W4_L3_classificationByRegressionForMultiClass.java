package weka_2nd;

import java.io.*;
import java.util.Random;
import weka.classifiers.*;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.rules.OneR;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.MakeIndicator;

public class W4_L3_classificationByRegressionForMultiClass {

	public static void main(String args[]) throws Exception{
		W4_L3_classificationByRegressionForMultiClass obj = new W4_L3_classificationByRegressionForMultiClass();
		String fileName= "iris";
		System.out.println(fileName + " : ");
		
		/*****************************************************************************
		 *  LinearRegression ������ ����
		 *  https://svn.cms.waikato.ac.nz/svn/weka/branches/stable-3-8/weka/lib/ �����Ͽ�
		 *  arpack_combined.jar, mtj.jar, core.jar �� �ܺ� jar �� ����Ʈ �ؾ� �Ѵ�.
		 ******************************************************************************/
		obj.irisRegressionForMakeIndicatorFilter(fileName,new LinearRegression(), "last"); // versinica
		obj.irisRegressionForMakeIndicatorFilter(fileName,new LinearRegression(), "2");    // versicolor
		obj.irisRegressionForMakeIndicatorFilter(fileName,new LinearRegression(), "1");    // setosa
 
	}

	public void irisRegressionForMakeIndicatorFilter(String fileName, Classifier model,String valueIndices) throws Exception{
		int seed = 1;
		int numfolds = 10;
		int numfold = 0;	
		// 1) data loader 
		Instances data=new Instances(new BufferedReader(new FileReader("D:\\Weka-3-9\\data\\"+fileName+".arff")));
		data.setClassIndex(data.numAttributes()-1); // ���͸��� ���� Ŭ���� ����
		/*****************************
		 * MakeIndicator ���� ���� ����
		 *****************************/
		MakeIndicator filter = new MakeIndicator(); 
		filter.setValueIndices(valueIndices);
		filter.setInputFormat(data);
		data = Filter.useFilter(data, filter);
		/*****************************
		 * MakeIndicator ���� ���� ����
		 *****************************/
		Instances train = data.trainCV(numfolds, numfold, new Random(seed));
		Instances test  = data.testCV (numfolds, numfold);
		
		// 2) class assigner
		train.setClassIndex(train.numAttributes()-1);
		test. setClassIndex(test. numAttributes()-1);
		
		// 3) cross validate setting  
		Evaluation eval=new Evaluation(train);
//		Classifier model=classifier; // �Ű��������� ���� ������ model��ü�� ���� ���	
		
		// 4) model run 
		model.buildClassifier(data);
		eval.crossValidateModel(model, train, numfolds, new Random(seed));
		
		// 5) evaluate
		eval.evaluateModel(model, test);
		
		// 6) print Result text
		System.out.println("\n******************************************************");
		System.out.println("    model for " + filter.getValueIndices());
		System.out.println("******************************************************");
		System.out.println(model.toString() +"\n"+eval.toSummaryString()); // ȸ�ͺм��� ���з��� ���� �������� ���� ȸ�͹����� ������ ��ǥ�� �߿�
		
		// 7) �Ӱ��� ����
		System.out.println(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> �Ӱ��� ���� >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
		this.makeCriticalPoint(data);
		System.out.println("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< �Ӱ��� ���� <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
		
		
	}
	
	public void makeCriticalPoint(Instances data) throws Exception{
		
		W4_L3_classificationByRegressionFor2valueClass obj = new W4_L3_classificationByRegressionFor2valueClass();


		// 2) AddClassification,  NumericToNominal, Remove ���� ����
		data = obj.applyFilters(new LinearRegression(), data, "5", "1-4");
		
		// 3) OneR �з��� ���� �Ӱ��� ����
		obj.diabeteOneRForAddclassificationFilter(new OneR(),data,100);// ������ �غ� ���� minBucketSize Ȯ��  
	}
}
