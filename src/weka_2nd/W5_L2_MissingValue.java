package weka_2nd;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.io.*;
import java.util.ArrayList;
import java.util.Random;

import javax.swing.JFrame;

import weka.classifiers.*;
import weka.classifiers.trees.J48;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

public class W5_L2_MissingValue {

	ArrayList<Integer> attrIndexWithMissingValue = new ArrayList<Integer>();

	public static void main(String args[]) throws Exception{
		  W5_L2_MissingValue obj = new W5_L2_MissingValue();
		  String fileName= "labor";
		  System.out.println(fileName + " : ");

		  obj.missingValue(fileName,new J48(), false, false);   // 1) ���ͳ� �Ӽ� ���� ���� �����ϰ� �з�
		  obj.missingValue(fileName,new J48(), false, true);    // 2) �Ӽ� ���� ���� + 3) �з�
		  obj.missingValue(fileName,new J48(), true, false);    // 4) ���͸� ���� + 5) �з�
	}
	
	public void missingValue(String fileName, Classifier model,boolean applyReplaceMissingValueFilter, boolean eraseMissingValue) throws Exception{
		  int numfolds = 10;
		  int numfold = 0;
		  int seed = 1;
		  
		  // 1) data loader 
		  Instances data=new Instances(new BufferedReader(new FileReader("D:\\Weka-3-9\\data\\"+fileName+".arff")));
		  data.setClassIndex(data.numAttributes()-1); 
		  
		  if(eraseMissingValue) {  
			  // ������ 33% �̻� �Ӽ� ����
			  data = this.deleteAttributeWithMissingValue(model, data);  		
		  }else{
			  // ReplaceMissingValueFilter ���� ���� ���ο� ���� �����ͼ�Ʈ ��ȯ (���ڴ� ���, ������� mode ������ ��ü)		  
			  data = this.applyReplaceMissingValueFilter(applyReplaceMissingValueFilter, model, data);
		  }
		  
		  // ������ ���� ��� �� ���� �Լ�
		  double totalMissingCount = this.missingCount(data);
		  
		  Instances train = data.trainCV(numfolds, numfold, new Random(seed));
		  Instances test  = data.testCV (numfolds, numfold);

		  // 2) class assigner
		  train.setClassIndex(train.numAttributes()-1);
		  test.setClassIndex(test.numAttributes()-1);
		  
		  // 3) cross validate setting  
		  Evaluation eval=new Evaluation(train);
		  eval.crossValidateModel(model, data, numfolds, new Random(seed));
		  
		  // 4) model run 
		  model.buildClassifier(train);
		  
		  // 5) evaluate
		  eval.evaluateModel(model, test);
		  
		  // 6) print Result text
		  this.printResultTitle(model, applyReplaceMissingValueFilter, eraseMissingValue);
		  System.out.println(model.toString() +"\n"+eval.toSummaryString()); 
		  
		  //7) view tree model
		  this.treeVeiwInstances(data, totalMissingCount, (J48)model, eval, applyReplaceMissingValueFilter, eraseMissingValue);
	}

    /******************************
     * ReplaceMissingValues ���� ���� �Լ�
     *****************************/
	public Instances applyReplaceMissingValueFilter(boolean applyReplaceMissingValueFilter, Classifier model, Instances data) throws Exception{
		if(!applyReplaceMissingValueFilter) return data;
		ReplaceMissingValues filter = new ReplaceMissingValues();
		data.setClassIndex(data.numAttributes()-1); // class assigner
		filter.setInputFormat(data);
		data = Filter.useFilter(data, filter);
		System.out.println("\n****************************************************************************");
		System.out.println("\t\t 4) ReplaceMissingValue applied ");
		System.out.println("****************************************************************************");

		return data;
	}
		 
	 /*************************************************************************************************
	  * 10�� �Ӽ� ���� ���� ( GUI �Ǵ� missingCount �Լ� ���� ������ 33% �̻� �Ӽ� index �ĺ��� �� �ִ�.)
	  *************************************************************************************************/
	 public Instances deleteAttributeWithMissingValue(Classifier model, Instances data) throws Exception{
		Remove filter = new Remove();

//		filter.setAttributeIndices("4,5,7-10,13-16");
		String s_index_array = "";
		for (Integer index : attrIndexWithMissingValue) s_index_array += index+",";
		filter.setAttributeIndices(s_index_array); // ���� �޾� �־ ���� ����
		data.setClassIndex(data.numAttributes()-1); // class assigner
		filter.setInputFormat(data);
		data = Filter.useFilter(data, filter);
		System.out.println("\n****************************************************************************");
		System.out.println("\t\t 2) delete " + attrIndexWithMissingValue.size() + " attribute ");
		System.out.println("****************************************************************************");
		System.out.println("deleted index =" + s_index_array); 

		return data;
	 }

	 /*************************************************************
	  * ������ 33% ���� �� 33% �̻� ������ ���� �Ӽ� index ���� (�ִ�������� 33.6%��)
	  ************************************************************/
	 public double missingCount(Instances data) throws Exception {
		 double totalMissingCount = 0.0;
		 double[] missingCountByAttr = new double[data.numAttributes()];
		 for(int x=0 ; x < data.size() ; x++){
			 Instance row = data.get(x);
			 for(int y=0 ; y < data.numAttributes(); y++){
				 try{
//					 System.out.print(row.stringValue(y) + " , ");
				 }catch (java.lang.IllegalArgumentException iia){
//					 System.out.print(row.value(y) + " , ");
				 }
				 if( row.isMissing(y) ) {
					 totalMissingCount++;
					 missingCountByAttr[y]++; 
				 }
			 }
//			 System.out.println("");
		 }
		 
		 double totalDataCount = data.size() * data.numAttributes();		 
		 double missingRatio = totalMissingCount / totalDataCount;

		 System.out.println("\n--------------------------------------------------------------------------");
		 System.out.println("\t\t Missing ratio : " + String.format("%.2f",missingRatio * 100) + " %" );
		 System.out.println("--------------------------------------------------------------------------");
		 for(int y=0 ; y < data.numAttributes(); y++){
//			 System.out.println("y = " + y + " = " + missingCountByAttr[y]);
			 if(missingCountByAttr[y] / data.size() > 0.33){ // missingRatio = 0.33
//				 System.out.println("deleted attr index = " + y);
				 attrIndexWithMissingValue.add((y+1));
			 }
		 }			
		 
		 return totalMissingCount;
	 }	 
	 
	 /**************************
	  * weka ���� �ð�ȭ (treeView)
	  **************************/
	 public void treeVeiwInstances(Instances data, double totalMissingCount, J48 tree, Evaluation eval, 
			                       boolean applyReplaceMissingValueFilter, boolean eraseMissingValue) throws Exception {

		 double missingRation = totalMissingCount/(data.size() * data.numAttributes()) * 100;
		 String graphName = "";
		 if(applyReplaceMissingValueFilter){
			 graphName= "4) ReplaceMissingValue applied";		
		 }else{
			 if(eraseMissingValue){
				 graphName= "3)" + attrIndexWithMissingValue.size() + " attribute deleted J48";		
			 }else{
				 graphName= "1) no filterd, no attr deleted";		
			 }
		 }		 
		 graphName += " , ������ = " + String.format("%.2f",missingRation) + " %";
		 graphName += " , ���з��� = " + String.format("%.2f",eval.pctCorrect()) + " %";
	     TreeVisualizer panel = new TreeVisualizer(null,tree.graph(),new PlaceNode2());
	     JFrame frame = new JFrame(graphName);
	     frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	     frame.getContentPane().setLayout(new BorderLayout());
	     frame.getContentPane().add(panel);
	     frame.setSize(new Dimension(800,500));
	     frame.setLocationRelativeTo(null);
	     frame.setVisible(true);
	     panel.fitToScreen();
	     System.out.println("See the " + graphName + " plot");
	 }     

	 /**************************
	  * ��� ��� title ����
	  **************************/
	 public void printResultTitle (Classifier model,boolean applyReplaceMissingValueFilter, boolean eraseOutlier){
		  System.out.println("\n****************************************************************************");
		  if(!eraseOutlier){
			  System.out.println("\t\t " + ((!applyReplaceMissingValueFilter)?"1) no filterd, no attr deleted ":"5) filtered J48"));
		  }else{
			  System.out.println("\t\t 3) " + attrIndexWithMissingValue.size() + " attribute deleted J48");
		  } 
		  System.out.println("****************************************************************************");
	 }
}
