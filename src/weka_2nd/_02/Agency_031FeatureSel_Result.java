package weka_2nd._02;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeSelection;
import weka.core.Instances;

public class Agency_031FeatureSel_Result  {

	public ASEvaluation attrEval = null;
	public double[] ranked = null;
	public int[] index = null;
	public String[] attrName = null;
	public int[] weight = null;
	
	public Agency_031FeatureSel_Result (int length){
		this.ranked = new double [length];
		this.index = new int [length];
		this.attrName = new String [length];
		this.weight = new int [length];		
	}
	
	public void setAttributeAlogrithms(ASEvaluation attrEval) throws Exception{
		this.attrEval = attrEval;
	}
	
	public void setResult(AttributeSelection attrSelector, Instances data) throws Exception{

		int[] selectedSEQ = attrSelector.selectedAttributes();
		
		// 선별된 속성 순서별 출력
		for (int i=0 ; i < selectedSEQ.length -1; i++) {			
			String attrName= data.attribute(selectedSEQ[i]).name();
			
			this.attrName[i] = attrName;
			this.index[i] = selectedSEQ[i]+1;
			this.weight[i] = 5 - (i+1);
			
//			System.out.println(MoreWekaCommon.getFeatureAlgorithmName(this.attrEval) + 
//					           ", index = " + index[i] + 
//			                   ", attr. Name: " + attrName + 
//			                   ", weight : " + weight[i]
//			                   );
		}
		
		System.out.println("");
	}
	
}
