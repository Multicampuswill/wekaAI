package base;

import java.io.*;
import java.util.ArrayList;
import java.util.Enumeration;
import weka.classifiers.Classifier;
import weka.core.*;
import weka.core.converters.CSVLoader;

public class SimpleWeka1_titanic_model_load_val2 {

	public static void main(String args[]) throws Exception {
		System.out.println("하나의 데이터를 넣어, 판단해보자.!!");
		// arff파일 형식을 만들어주어야 한다.
		ArrayList attributes = new ArrayList();// attribute 목록
		ArrayList aliveVal = new ArrayList(); // 타겟값 목록
		Instances dataRaw = null; // 확인할 데이터 객체

		System.out.println("=======attr리스트 얻기==========");
		CSVLoader csvLoader = new CSVLoader();
		csvLoader.setSource(new File("data/titanic2_pre.csv"));

		Instances file = csvLoader.getDataSet();
		System.out.println(file);
		// Instance extract = file.
		Enumeration<Attribute> attr_list = file.enumerateAttributes();
		System.out.println(attr_list);
		while (attr_list.hasMoreElements()) {
			attributes.add(attr_list.nextElement());
		}
		System.out.println(attributes);
		dataRaw = new Instances("TestInstances", attributes, 0);
		dataRaw.setClassIndex(dataRaw.numAttributes() - 1);

		System.out.println();
		System.out.println("======== 여기까지 arff만든 부분 >>\n" + dataRaw + "========");

		// double[] instanceValue1 = { 0, 24, 0, 8.05, 1, 0, 0 }; 
		// double[] instanceValue1 = {0,24,0,8.05,1,0,0}; //0
		// double[] instanceValue1 = { 1, 38, 1, 71.2833, 0, 1, 0};//1
		double[] instanceValue1 = { 0, 40, 0, 27.7208, 0, 1, 0, 0 };// 0
		dataRaw.add(new DenseInstance(1.0, instanceValue1));

		System.out.println();
		System.out.println("======== 여기까지 arff만든 부분 >>\n" + dataRaw + "\n========\n");
		System.out.println(dataRaw.firstInstance());
		Classifier cls = (Classifier) SerializationHelper.read("model/titanic_model.model");
		System.out.println(cls.classifyInstance(dataRaw.firstInstance()));

		System.out.println("=========================================");
		// System.out.println(file);
	}

}
