package weka_2nd._02;



import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CorrelationAttributeEval;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.OneRAttributeEval;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.WrapperSubsetEval;
import weka.core.Instances;

public class Agency_030SelectFeature {

	private ASEvaluation[] attrEval = null;
	private ASSearch[] searchMethod = null;
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}
	
	public Agency_030SelectFeature (){
		this.attrEval = new ASEvaluation[5];

		this.attrEval[0] = new WrapperSubsetEval();
		this.attrEval[1] = new CorrelationAttributeEval();
		this.attrEval[2] = new GainRatioAttributeEval();
		this.attrEval[3] = new InfoGainAttributeEval();
		this.attrEval[4] = new OneRAttributeEval();

		this.searchMethod = new ASSearch[2];
		this.searchMethod[0] = new BestFirst();
		this.searchMethod[1] = new Ranker();
	}


	 // 속성선택
	public Agency_031FeatureSel_Result[] attrSelFeature(Instances data) throws Exception{
		Agency_031FeatureSel_Result[] result = null;
				
		// 목표변수 설정
		data.setClassIndex(data.numAttributes()-1);
		
		// 속성선택 패널객체 생성
		AttributeSelection attrSelector = new AttributeSelection();

		result = new Agency_031FeatureSel_Result[this.attrEval.length]; // WrapperSubsetEval 제외
		
		for(int x=0 ; x < this.attrEval.length ; x++){
			result[x] = new Agency_031FeatureSel_Result(data.numAttributes()); 
				
			// 선택알고리즘 
			attrSelector.setEvaluator(attrEval[x]);			
			
			// 탐색방법
			if(x == 0)
				attrSelector.setSearch(searchMethod[0]); // BestFirst
			else
				attrSelector.setSearch(searchMethod[1]); // Ranker
					
			// 속성선택 실시
			attrSelector.SelectAttributes(data);
				
			// 선별된 속성 결과 저장 및  출력
			result[x].setAttributeAlogrithms(attrEval[x]);
			result[x].setResult(attrSelector, data);
		}	
		return result;
	}
}
