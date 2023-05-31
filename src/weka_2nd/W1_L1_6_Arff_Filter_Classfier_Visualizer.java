package weka_2nd;

import java.io.*;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SimpleLogistic;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.*;
import weka.core.*;

public class W1_L1_6_Arff_Filter_Classfier_Visualizer {
	
	public static void main(String args[]) throws Exception{
		W1_L1_6_Arff_Filter_Classfier_Visualizer obj = new W1_L1_6_Arff_Filter_Classfier_Visualizer();
		System.out.println("���� ������");
		obj.RemoveWithValues(false);
		System.out.println("���� RemoveWithValues ����");
		obj.RemoveWithValues(true);
	
	}

	public void RemoveWithValues(boolean isFilter) throws Exception{
		int numfolds = 10;
		int numfold = 0;
		int seed = 1;
		// 1) data loader (�Ʒ� �����Ϳ� �ؽ�Ʈ �����͸� �⺻ 8:2 �� �и��Ѵ�.)
		Instances data=new Instances(
				       new BufferedReader(
				       new FileReader("D:\\Weka-3-9\\data\\labor.arff")));
		/**********************************************************
		 * 1-1) ���� ���� ����
		 **********************************************************/
		if(isFilter){
			RemoveWithValues filter = new RemoveWithValues();
//			filter.setOptions(Utils.splitOptions("-S 0.0 -C 5 -L 1"));
			filter.setAttributeIndex("5");
			filter.setNominalIndices("1");
			filter.setInputFormat(data);
			data = Filter.useFilter(data, filter);		
		}	
		/**********************************************************
		 * 1-1) ���� ���� ����
		 **********************************************************/
		
		Instances train = data.trainCV(numfolds, numfold, new Random(seed));
		Instances test  = data.testCV (numfolds, numfold);

		// 2) class assigner
		train.setClassIndex(train.numAttributes()-1);
		test.setClassIndex(test.numAttributes()-1);
		
		// 3) cross validate setting  
		Evaluation eval=new Evaluation(train);
		Classifier model=new SimpleLogistic();
		eval.crossValidateModel(model, train, numfolds, new Random(seed));

		// 4) model run 
		model.buildClassifier(train);
		
		// 5) evaluate
		eval.evaluateModel(model, test);
		
		// 6) print Result text
		System.out.println(eval.toSummaryString()); // === Evaluation result ===
	}
}
