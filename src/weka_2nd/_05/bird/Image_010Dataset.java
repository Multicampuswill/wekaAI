package weka_2nd._05.bird;

import java.io.File;

import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka_2nd.MoreWekaCommon;

public class Image_010Dataset {

	private Instances imgList= null;
	private Instances filteredList = null;
	private Classifier model = null;
	
	public static void  main (String args[]) throws Exception{
		Image_010Dataset obj = new Image_010Dataset();
		obj.openArffonWeb();
		obj.preprocess();
		obj.build();
		obj.classify();
	}
	
	// 1) data loader
	public void openArffonWeb() throws Exception{
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File(MoreWekaCommon.arffPath
				               +"\\image\\bird\\butterfly_vs_owl.arff"));
		imgList = loader.getDataSet();
		System.out.println(" 전체 데이터 건 수 : " + imgList.size());
	}
	
	// 2) 전처리 (각종 필터링)
	public void preprocess() throws Exception{
		Image_020Preprocess preprocess = new Image_020Preprocess(this.imgList);
		this.setFilteredList(preprocess.action());
	}
	

	// 3) 모델 학습/평가/저장
	public void build() throws Exception{
		Image_030PerceptronClassifier imageClassfy 
		        = new Image_030PerceptronClassifier(new MultilayerPerceptron());
		this.model = imageClassfy.buildModel(this.imgList);
	}
	
	// 4) 모델 분류
	public void classify() throws Exception{
		Image_040Classify classify 
        = new Image_040Classify(this.model);

		System.out.println("**********************");
		System.out.println("문장 학습 + 단어 분류");
		System.out.println("**********************");
		classify.useModel(this.setValidateData());
		
	}
	
	public Instances setValidateData() throws Exception{
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File(MoreWekaCommon.arffPath
				       +"\\image\\bird\\butterfly_vs_owl_from_internet.arff"));
	    Instances validateData =  loader.getDataSet();
		System.out.println(" 전체 데이터 건 수 : " + validateData.size());
		Image_020Preprocess preprocess = new Image_020Preprocess(validateData);
		preprocess.action();
	    
	    return validateData;
	}

	public Instances getFilteredList() {
		return filteredList;
	}

	public void setFilteredList(Instances filteredList) {
		this.filteredList = filteredList;
	}
}
