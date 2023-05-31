package base;

import java.io.*;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.*;
import weka.core.converters.CSVLoader;

public class SimpleWeka1_titanic_model_load_val {

	public static void main(String args[]) throws Exception {
		int numfolds = 10;
		int numfold = 0;
		int seed = 1;
		// 1) data loader
		// Instances data = new Instances(new BufferedReader(new
		// FileReader("data/titanic2_pre.arff")));
		CSVLoader csvLoader = new CSVLoader();
		csvLoader.setSource(new File("data/titanic2_pre.csv"));

		Instances data = csvLoader.getDataSet();

		Instances train = data.trainCV(numfolds, numfold, new Random(seed));
		Instances test = data.testCV(numfolds, numfold);

		RandomForest model = new RandomForest();
		System.out.println("model>> " + model);

		// 2) class assigner
		train.setClassIndex(train.numAttributes() - 1);
		test.setClassIndex(test.numAttributes() - 1);

		//////////////////////////////////
		Evaluation eval = new Evaluation(train);

		eval.crossValidateModel(model, train, numfolds, new Random(seed));

		// 4) random forest run
		model.buildClassifier(train);

		// 5) evaluate
		eval.evaluateModel(model, test);

		// 하나만 테스트해보자.
		// Instance instance = test.firstInstance();
//		Instance instance = test.get(2);
//		System.out.println(instance);
//		System.out.println(instance.classValue());
//		System.out.println(instance.classIndex());
//		double result = eval.evaluateModelOnce(model, instance);
//		System.out.println(result);

		// 6) print Result text
		System.out.println(model); // model info
		System.out.println(eval.toSummaryString()); // === Evaluation result ===
		System.out.println(eval.toMatrixString()); // === Confusion Matrix ===

		System.out.println("======================================");
		System.out.println("하나의 데이터를 넣어, 판단해보자.!!");

		// arff파일 형식을 만들어주어야 한다.
		ArrayList attributes = new ArrayList();// attribute 목록
		ArrayList aliveVal = new ArrayList(); // 타겟값 목록
		Instances dataRaw = null; // 확인할 데이터 객체

		System.out.println("=======attr리스트 얻기==========");
		Enumeration<Attribute> attr_list = data.enumerateAttributes();
		while (attr_list.hasMoreElements()) {
			attributes.add(attr_list.nextElement());
		}
		System.out.println(attributes);

//		aliveVal.add("no"); // target값 지정
//		aliveVal.add("yes"); // target값 지정
//
//		attributes.add(new Attribute("alive", aliveVal));
		dataRaw = new Instances("TestInstances", attributes, 0);
		dataRaw.setClassIndex(dataRaw.numAttributes() - 1);

		System.out.println();
		System.out.println("======== 여기까지 arff만든 부분 1>>\n" + dataRaw + "========");

		// double[] instanceValue1 = {0,24,0,8.05,1,0,0}; //0
		// double[] instanceValue1 = { 1, 38, 1, 71.2833, 0, 1, 0};//1
		double[] instanceValue1 = { 0, 40, 0, 27.7208, 0, 1, 0, 0 };// 0
		dataRaw.add(new DenseInstance(1.0, instanceValue1));

		System.out.println();
		System.out.println("======== 여기까지 arff만든 부분 2>>\n" + dataRaw + "\n========\n");
		System.out.println(dataRaw.firstInstance());

		Classifier cls = (Classifier) SerializationHelper.read("model/titanic_model.model");
		System.out.println(cls.classifyInstance(dataRaw.firstInstance()));

		System.out.println("=========================================");
		// System.out.println(test);
	}
}


