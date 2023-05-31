package weka_2nd;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.io.*;
import java.util.ArrayList;
import java.util.Random;

import javax.swing.JFrame;

import weka.classifiers.*;
import weka.classifiers.functions.LeastMedSq;
import weka.classifiers.functions.LinearRegression;
import weka.core.*;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AddClassification;

public class W5_L1_OutlierWithCSV {

	public W5_L1_OutlierWithCSV() {
		// TODO Auto-generated constructor stub
	}

	public static void main(String args[]) throws Exception{
		  W5_L1_OutlierWithCSV obj = new W5_L1_OutlierWithCSV();
		  String fileName= "regression_outliers";
		  System.out.println(fileName + " : ");

		  // http://www.java2s.com/Code/Jar/l/Downloadleastmedsquared101jar.htm ���� jar ���� �ٿ�ε��� �������� �� �ܺ� jar ����
		  obj.outlierWithCSV(fileName,new LinearRegression(), false, false);   // 1) �̻� �ĺ�   + 2) ������
		  obj.outlierWithCSV(fileName,new LinearRegression(), true, false);    // 3) AddClassification ���� ���� + 4) ����ȸ�ͽ� ����
		  obj.outlierWithCSV(fileName,new LeastMedSq(),       true, false);    // 5) LeastMedSq ���� 
		  obj.outlierWithCSV(fileName,new LinearRegression(), true, true);     // 6) 8�� �̻� ������ ����ȸ�ͽ� ����
	}
	
	public void outlierWithCSV(String csvFileName, Classifier model,boolean applyAddClassificationFilter, boolean eraseOutlier) throws Exception{
		  int numfolds = 10;
		  int numfold = 0;
		  int seed = 1;
		  
		  // 1) csv data loader 
		  CSVLoader csvloader = new CSVLoader();
		  csvloader.setSource(new File("D:\\Weka-3-9\\data\\"+csvFileName+".csv"));
		  Instances data = csvloader.getDataSet(); 
		  
		  if(!eraseOutlier) { 
			  // AddClassification ���� ���� ���ο� ���� �����ͼ�Ʈ ��ȯ (8�� �̻� ���� ��� �̿ܿ��� ����)			  
			  data = this.applyAddClassificationFilter(applyAddClassificationFilter, model, data);		
		  }else{ 
			  // �̻� �������ο� ���� �����ͼ�Ʈ ��ȯ (8�� �̻� ���� ��쿡�� ����)
			  data = this.eraseOutliner(model, data);  
		  }
		  
		  Instances train = data.trainCV(numfolds, numfold, new Random(seed));
		  Instances test  = data.testCV (numfolds, numfold);

		  // 2) class assigner
		  train.setClassIndex(train.numAttributes()-1);
		  test.setClassIndex(test.numAttributes()-1);
		  
		  // 3) cross validate setting  
		  Evaluation eval=new Evaluation(train);
		  
		  // 4) model run 
		  model.buildClassifier(train);
		  eval.crossValidateModel(model, train, numfolds, new Random(seed));
		  
		  // 5) evaluate
		  eval.evaluateModel(model, test);
		  
		  // 6) print Result text
		  this.printResultTitle(model, applyAddClassificationFilter, eraseOutlier);
		  System.out.println(model.toString() +"\n"+eval.toSummaryString()); 
	}

    /******************************
     * AddClassification ���� ���� �Լ�
     *****************************/
	public Instances applyAddClassificationFilter(boolean applyAddClassificationFilter, Classifier model, Instances data) throws Exception{
		if(model instanceof LeastMedSq) return data;
		if( !applyAddClassificationFilter ){
			System.out.println("\n****************************************************************************");
			System.out.println("\t\t 1) outlier recognition");
			System.out.println("****************************************************************************");
			this.plot2DInstances(data, "1) outlier recognition",1);
		}else{	   
			AddClassification filter = new AddClassification();
			filter.setClassifier(model);
			filter.setOutputClassification(true);
			data.setClassIndex(data.numAttributes()-1); // class assigner
			filter.setInputFormat(data);
			data = Filter.useFilter(data, filter);
			System.out.println("\n****************************************************************************");
			System.out.println("\t\t 3) data linearization ");
			System.out.println("****************************************************************************");
			this.plot2DInstances(data, "3) data linearization",2);
		}	
		return data;
	}
		 
	 /**************************
	  * 8�� ���� ���� (63�� ~ 70��)
	  **************************/
	 public Instances eraseOutliner(Classifier model, Instances data) throws Exception{
		 if(model instanceof LeastMedSq) return data;
		 // ���ο� �����ͼ�Ʈ ����
		 ArrayList<Attribute> attr = new ArrayList<Attribute>();
		 attr.add(new Attribute("year"));
		 attr.add(new Attribute("phone calls"));
		 Instances erasedData = new Instances("ErasedData", attr,0) ;
		  
		 // 63~70 �� �����Ϳ��� �ν��Ͻ��� ���ο� �����ͼ�Ʈ�� ����
		 for (int i = 0; i < data.size(); i++) {
			 Instance instance = data.get(i);
			 int year = (int) instance.value(0);
			 if(63 <= year && year <=70){
//				 System.out.println(i + " , year = " + year + " erased ");
			 }else{
				 erasedData.add(instance);
			 }
		 } 
		 this.plot2DInstances(erasedData, "erased Data",1); 
		 return erasedData;
	 }

	 /**************************
	  * weka ���� �ð�ȭ (plot2D)
	  **************************/
	 public void plot2DInstances(Instances data, String graphName, int yIndex) throws Exception {

	     weka.gui.visualize.Plot2D panel = new weka.gui.visualize.Plot2D();
	     panel.setInstances(data);    
	     panel.setXindex(0);     
	     panel.setYindex(yIndex); // AddClassification ���� ������ �Ӽ��� �߰��Ǳ⿡ index = 2 �� �����ؾ� �� (�׿ܴ� index = 1)
	     panel.setCindex(data.numAttributes() - 1);

	     JFrame frame = new JFrame(graphName);
	     frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	     frame.getContentPane().setLayout(new BorderLayout());
	     frame.getContentPane().add(panel);
	     frame.setSize(new Dimension(600, 400));
	     frame.setLocationRelativeTo(null);
	     frame.setVisible(true);
	     System.out.println("See the " + graphName + " plot");
	 }     

	 /**************************
	  * ��� ��� title ����
	  **************************/
	 public void printResultTitle (Classifier model,boolean applyAddClassificationFilter, boolean eraseOutlier){
		  System.out.println("\n****************************************************************************");
		  if(!eraseOutlier){
			  if( !(model instanceof LeastMedSq) )
				   System.out.println("\t\t " + ((!applyAddClassificationFilter)?"2) Correlation coefficient":"4) LinearRegression Model Fomular"));
			  else 
				   System.out.println("\t\t 5) LeastMedSq Correlation coefficient");
		  }else{
			  System.out.println("\t\t 6) outlier erased Correlation coefficient");
		  } 
		  System.out.println("****************************************************************************");
	 }
}
