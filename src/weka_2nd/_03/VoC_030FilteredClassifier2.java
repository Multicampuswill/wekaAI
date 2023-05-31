package weka_2nd._03;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.util.Random;

import javax.swing.JFrame;

import org.apache.commons.math3.stat.descriptive.AggregateSummaryStatistics;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.unsupervised.attribute.ClassAssigner;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;
import weka_2nd.MoreWekaCommon;

public class VoC_030FilteredClassifier2 {

	private FilteredClassifier model = null;
	private Classifier innerModel = null;

	public VoC_030FilteredClassifier2(){
		this.model = new FilteredClassifier();
		this.innerModel = new J48();
	}

	public VoC_030FilteredClassifier2(Classifier cls){
		if( cls instanceof J48)
			this.innerModel = new J48();
		else if( cls instanceof NaiveBayesMultinomial)
			this.innerModel = new NaiveBayesMultinomial();
			
		this.model = new FilteredClassifier();
	}
	
	public void saveModel(Classifier model) throws Exception{
		if(innerModel instanceof J48)
			weka.core.SerializationHelper.write(MoreWekaCommon.modelPath +"textMining\\voc_" +
					                            MoreWekaCommon.getModelName(model), model);
	}

	public Classifier buildModel(Instances train, Instances test) throws Exception{
		
		this.suppliedTestSet(train, test);		
		
		// 최적모델 저장
		this.saveModel(model);
		
		// 최적모델 반환
		return model;
	}
	
	public double suppliedTestSet(Instances train, Instances test) throws Exception{
		int numfolds = 10;
		  
		// 1) data loader 		
		
		// 2) class assigner
		
		// 3) set innerClassifer, filters
		model.setClassifier(new J48());
		this.setFilteredClassifier(model, train);
		  
		// 4) cross validate setting  
		Evaluation eval=new Evaluation(train);	
		eval.crossValidateModel(model, train, numfolds, new Random(10));	  

		// 5) model run 
		model.buildClassifier(train);
		   
		// 6) evaluate
		eval.evaluateModel(model, test);
		
		// 7) print Result text 
		this.printResult(eval , test);
				
		return eval.pctCorrect();
	}

	public Classifier buildModel(Instances data) throws Exception{
		
		this.crossValidataion(data);		
		
		// 최적모델 저장
		this.saveModel(model);
		
		// 최적모델 반환
		return model;
	}
	
	public double crossValidataion(Instances data) throws Exception{
		int numfolds = 10;
		int numfold = 0;
		  
		// 1) data loader 		
		Instances train = data.trainCV(numfolds, numfold, new Random(10));
		Instances test  = data.testCV (numfolds, numfold);
		
		// 2) class assigner
		train.setClassIndex(train.numAttributes()-1); 
		test.setClassIndex(test.numAttributes()-1);  
		
		// 3) set innerClassifer, filters
		model.setClassifier(innerModel);
		model = this.setFilteredClassifier(model,data);
		  
		// 4) cross validate setting  
		Evaluation eval=new Evaluation(train);	
		eval.crossValidateModel(model, train, numfolds, new Random(10));	  

		// 5) model run 
		model.buildClassifier(train);
		   
		// 6) evaluate
		eval.evaluateModel(model, test);
		
		// 7) print Result text 
		this.printResult(eval , test);
		
		if(innerModel instanceof J48)
			this.treeVeiwInstances(data, (J48)innerModel, eval);
				
		return eval.pctCorrect();
	}
	
	public void printResult(Evaluation eval, Instances test) throws Exception{
		
		System.out.println(MoreWekaCommon.getModelName(this.model.getClassifier()) + " : "  +
				   "분류대상 데이터 건 수 : " + (int)eval.numInstances() + 
		           ", 정분류 건수 : " + (int)eval.correct() + 
		           ", 정분류율 : " + String.format("%.1f",eval.correct() / eval.numInstances() * 100) +" %" +
		           ", \n 오분류표 : \n" +  eval.toMatrixString() +
		           ""
					); 	
	}
	
	/**
	 * common-math jar 다운로드 위치 : http://apache.mirror.cdnetworks.com/commons/math/binaries/
	 * **/
	public double aggregateValue(double[] sum){
		AggregateSummaryStatistics aggregate = new AggregateSummaryStatistics();
		SummaryStatistics sumObj = aggregate.createContributingStatistics();
		for(int i = 0; i < sum.length; i++)  
			sumObj.addValue(sum[i]); 

		System.out.println("평균 : " + String.format("%.1f",aggregate.getMean()) 
		             + " %, 편차 : " + String.format("%.1f",aggregate.getStandardDeviation()));
		
		return aggregate.getMean();
	}

	// FilteredClassifier 내부 필터 및 내부 분류알고리즘 설정
	public FilteredClassifier setFilteredClassifier(FilteredClassifier filterCls, Instances data) throws Exception{
		MultiFilter mFilter = new MultiFilter();
		Filter[] filters = new Filter[2];
				
		filters[0] = new VoC_021Filter(data).setString2WorkVector().getStw();

		ClassAssigner assingner = new ClassAssigner();
		assingner.setClassIndex("first");
		filters[1] = assingner;
				
		mFilter.setFilters(filters);
		
		filterCls.setFilter(mFilter);		
		
		return filterCls;
	}

	 /**************************
	  * weka 제공 시각화 (treeView)
	  **************************/
	 public void treeVeiwInstances(Instances data, J48 tree, Evaluation eval) throws Exception {

		 String graphName = "";
		 graphName += " 정분류율 = " + String.format("%.2f",eval.pctCorrect()) + " %";
	     TreeVisualizer panel = new TreeVisualizer(null,tree.graph(),new PlaceNode2());
	     JFrame frame = new JFrame(graphName);
	     frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	     frame.getContentPane().setLayout(new BorderLayout());
	     frame.getContentPane().add(panel);
	     frame.setSize(new Dimension(800,500));
	     frame.setLocationRelativeTo(null);
	     frame.setVisible(true);
	     panel.fitToScreen();
//	     System.out.println("See the " + graphName + " plot");
	 }     
}
