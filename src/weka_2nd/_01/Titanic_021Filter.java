package weka_2nd._01;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.RenameAttribute;
import weka.filters.unsupervised.attribute.RenameNominalValues;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.attribute.Standardize;
import weka.filters.unsupervised.attribute.StringToNominal;

public class Titanic_021Filter {	

	private Instances data = null;

	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

	public Titanic_021Filter (Instances data){
		this.data = data;
	}
	
	public Instances action() throws Exception{
		this.renameAttributes(); // 속성명을 한글로 치환
		this.numericToNominal("1"); // 숫자형 데이터를 명목형으로 변환
		this.renameValues();     // 데이터 값의 일부를 한글로 치환 
		this.remove("10,13,14"); // 결측률 33% 이상 속성 삭제 (단,구명정번호 속성은 제외) 
		this.reOrder("1,3-last,2");          // 속성재배열 (2번째 구조여부 속성을 맨 뒤로 배치)
		this.stringToNominal("2,7,10");  // String형 데이터를 명목형으로 변환 
		return this.data;
	}	
	
	public Instances normalize () throws Exception{
		Normalize normalize = new Normalize();
		runFilter(normalize);
		return this.data;
	}

	public Instances standardize () throws Exception{
		Standardize standardize = new Standardize();
		runFilter(standardize);
		return this.data;
	}
	
	/******************
	 * 속성명을 한글로 전환
	 ******************/
	public void renameAttributes () throws Exception{
		this.setRenameAttributeFiler("1", "객실등급");
		this.setRenameAttributeFiler("2", "구조여부");
		this.setRenameAttributeFiler("3", "성명");
		this.setRenameAttributeFiler("4", "성별");
		this.setRenameAttributeFiler("5", "나이");
		this.setRenameAttributeFiler("6", "동승_배우자_형제수");
		this.setRenameAttributeFiler("7", "동승_부모_자녀수");
		this.setRenameAttributeFiler("8", "ticket번호");
		this.setRenameAttributeFiler("9", "요금");
		this.setRenameAttributeFiler("10", "객실번호");
		this.setRenameAttributeFiler("11", "탑승위치");
		this.setRenameAttributeFiler("12", "구명정번호 ");
		this.setRenameAttributeFiler("13", "신장");
		this.setRenameAttributeFiler("14", "목적지");
	}
	
	public void setRenameAttributeFiler(String index, String Replace) throws Exception{
		RenameAttribute renameAttr = new RenameAttribute();
		renameAttr.setAttributeIndices(index);
		renameAttr.setReplace(Replace);
		this.runFilter(renameAttr);
	}
	
	public void renameValues () throws Exception{
		this.setRenameValueFilter("1","1:1등급,2:2등급,3:3등급");// NumericToNominal 변경후 필터링
		this.setRenameValueFilter("2","1:생존,0:사망");
		this.setRenameValueFilter("4","female:여성,male:남성");
	}
	
	public void setRenameValueFilter(String index, String Replacement) throws Exception{
		RenameNominalValues renameValue = new RenameNominalValues();		
		renameValue.setSelectedAttributes(index);
		renameValue.setValueReplacements(Replacement);
		this.runFilter(renameValue);
	}

	public void numericToNominal(String index) throws Exception{
		NumericToNominal numToNom = new NumericToNominal();	
		numToNom.setAttributeIndices(index);
		this.runFilter(numToNom);	
	}

	public void stringToNominal(String index) throws Exception{
		StringToNominal strToNom = new StringToNominal();		
		strToNom.setAttributeRange(index);
		this.runFilter(strToNom);
	}
	
	public void reOrder(String index) throws Exception{
		Reorder strToNom = new Reorder();		
		strToNom.setAttributeIndices(index);
		this.runFilter(strToNom);
	}

	public Instances remove(String index) throws Exception{
		Remove remove = new Remove();		
		remove.setAttributeIndices(index);
		this.runFilter(remove);
		return this.data;
	}
	
	public Instances runFilter(Filter filter) throws Exception {
		filter.setInputFormat(data);	
		data = Filter.useFilter(data, filter);	
		return this.data;
	}
	
	public Instances getData() {
		return this.data;
	}
}
