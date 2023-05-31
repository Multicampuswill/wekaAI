package weka_2nd._01;

import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class Titanic_010Dataset {
	
	private Instances data = null;
	private Instances titanic_co_reorder = null;
	private Instances titanic_co_reorder_norm = null;
	private Instances titanic_co_reorder_stan = null;
	private Classifier model = null;
	
	public static void  main (String args[]) throws Exception{
		Titanic_010Dataset obj = new Titanic_010Dataset();
		obj.openArffonWeb();
		obj.preprocess();
		obj.experiment();
		obj.classify();
	}

	// 1) data loader
	public void openArffonWeb() throws Exception{
		ArffLoader loader = new ArffLoader();
		loader.setURL("https://www.openml.org/data/download/16826755/phpMYEkMl.arff");		
		this.data = loader.getDataSet();
		System.out.println("전체 데이터 건 수 : " + data.size());
	}
	
	// 2) 전처리 (각종 필터링)
	public void preprocess() throws Exception{
		Titanic_020Preprocess preprocess 
		         = new Titanic_020Preprocess(this.data);
		
		preprocess.filter();
		
		this.titanic_co_reorder = preprocess.getReorder();
		this.titanic_co_reorder_norm = preprocess.getNomalize();
		this.titanic_co_reorder_stan = preprocess.getStandardize();
	}
	

	// 3) 모델 학습/평가/저장
	public void experiment() throws Exception{
		Titanic_030Experiment experiment = new Titanic_030Experiment();
		this.model = experiment.experimentModel(this.titanic_co_reorder_stan);
	}
	
	// 4) 모델 분류
	public void classify() throws Exception{
		Titanic_04Classify classify 
        = new Titanic_04Classify(this.model);

		System.out.println("**********************");
		System.out.println("표준화 학습 + 속성재배열 분류");
		System.out.println("**********************");
		classify.useModel(this.titanic_co_reorder);

		System.out.println("**********************");
		System.out.println("표준화 학습 + 정규화 분류");
		System.out.println("**********************");
		classify.useModel(this.titanic_co_reorder_norm);
		
	}
	
}
