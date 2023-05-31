package weka_2nd._02;

import org.apache.commons.math3.stat.descriptive.AggregateSummaryStatistics;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

import weka.core.Instances;
import weka.filters.unsupervised.attribute.AddExpression;
import weka.filters.unsupervised.attribute.Copy;
import weka.filters.unsupervised.attribute.MergeManyValues;
import weka_2nd._01.Titanic_021Filter;

public class Agency_022Filter extends Titanic_021Filter {	

	public Agency_022Filter(Instances data) {
		super(data);
	}
	
	public Instances filter() throws Exception{
		
		super.remove("3,4"); // RANK, 법인별 평균속도 속성 삭제
		this.copy("3"); // platform 구분 위한 화면명 속성 복사
		this.mergeManyValues(); // 화면명 순서별 platform 으로 머징
		this.addExpression("ifelse(a4>",",0,1)", "isFast"); // 화면속도에 따른 빠름/느림 조건문
		super.numericToNominal("last"); // 0,1 을 명목형으로 변환
		this.renameValues("last", "1:fast,0:late");
		this.renameAttributes(); // 속성명들을 한글명으로 치환
		this.remove("4"); // 과적합 방지를 위해 화면속도 속성 삭제
		
		return super.getData();
	}	
	
	public Instances copy (String index) throws Exception{
		Copy copy = new Copy();
		copy.setAttributeIndices(index);
		return super.runFilter(copy);
	}
	
	public void mergeManyValues () throws Exception{
		this.setMergeManyValues("last", "platformB", "23-29");	
		this.setMergeManyValues("last", "platformA", "16-22");		
		this.setMergeManyValues("last", "platformC", "1-15");			
	}
	
	public Instances setMergeManyValues (String index, String label, String range) throws Exception {
		MergeManyValues mergeValues = new MergeManyValues();
		mergeValues.setAttributeIndex(index);
		mergeValues.setLabel(label);
		mergeValues.setMergeValueRange(range);
		return super.runFilter(mergeValues);
	}
	
	public Instances addExpression (String expression1, String expression2, String name) throws Exception {
		AddExpression addExp = new AddExpression();
		
		// 화면평균 산출
		double[] speed = new double [super.getData().numInstances()];
		for(int x=0 ; x < super.getData().numInstances() ; x++){
			speed[x] = super.getData().get(x).value(3); // 화면속도 누적
		}
		double speedMean = this.aggregateValue(speed); // 화면속도 평균 산출
		
		// ifelse(a4>5.388,0,1)"
		addExp.setExpression(expression1 + String.format("%.3f",speedMean)  + expression2);
		System.out.println("평균속도 : " + speedMean + " , 조건식 : " + addExp.getExpression());	
		addExp.setName(name);		
		return super.runFilter(addExp);		
	}
	
	public void renameValues(String index, String replace) throws Exception {
		super.setRenameValueFilter("last","1:fast,0:late");
	}

	public void renameAttributes () throws Exception{
		super.setRenameAttributeFiler("3", "화면명");
		super.setRenameAttributeFiler("4", "화면속도");
		super.setRenameAttributeFiler("5", "platform");
	}

	/**
	 * common-math jar 다운로드 위치 : http://apache.mirror.cdnetworks.com/commons/math/binaries/
	 * **/
	public double aggregateValue(double[] sum){
		AggregateSummaryStatistics aggregate = new AggregateSummaryStatistics();
		SummaryStatistics sumObj = aggregate.createContributingStatistics();
		for(int i = 0; i < sum.length; i++)  
			sumObj.addValue(sum[i]); 
			
		return aggregate.getMean();
	}	
}
