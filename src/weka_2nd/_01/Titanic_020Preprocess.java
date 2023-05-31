package weka_2nd._01;

import java.awt.Dimension;

import javax.swing.JFrame;
import javax.swing.JScrollPane;
import javax.swing.JTable;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.CorrelationAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instance;
import weka.core.Instances;

public class Titanic_020Preprocess {

	private Instances data = null;
	private Instances titanic_co_reorder = null;
	private Instances titanic_co_reorder_norm = null;
	private Instances titanic_co_reorder_stan = null;
	private Instances titanic_co_reorder_min = null;
	private Instances titanic_co_reorder_attr = null;
	
	private Titanic_021Filter filter = null;

	public Titanic_020Preprocess (Instances data){
		this.data = data;
	}

	public Instances getReorder(){return this.titanic_co_reorder;}
	public Instances getNomalize(){return this.titanic_co_reorder_norm;}
	public Instances getStandardize(){return this.titanic_co_reorder_stan;}
	public Instances getMin(){return this.titanic_co_reorder_min;}
	public Instances getSelectedAttr(){return this.titanic_co_reorder_attr;}
	
	public void filter() throws Exception{
		this.showRules("original data set",data);
		this.filter = new Titanic_021Filter(this.data);
		this.titanic_co_reorder = filter.action();
		this.showRules("속성 재배열", titanic_co_reorder);

		// 정규화
		this.titanic_co_reorder_norm = new Titanic_021Filter(titanic_co_reorder).normalize();
		this.showRules("속성 재배열 + 정규화", titanic_co_reorder_norm);

		// 표준화
		this.titanic_co_reorder_stan = new Titanic_021Filter(titanic_co_reorder).standardize();
		this.showRules("속성 재배열 + 표준화", titanic_co_reorder_stan);

		// 속성 임의 삭제
		this.titanic_co_reorder_min = new Titanic_021Filter(titanic_co_reorder).remove("2,7");
		this.showRules("속성 재배열 + 임의삭제", titanic_co_reorder_min);
		
		// 속성선택
		this.titanic_co_reorder_attr = this.attrSelClassifier(titanic_co_reorder);
		this.showRules("속성 재배열 + 속성선택", titanic_co_reorder_attr);
	}	

	// swing 테이블 내용 생성
	 public void showRules(String title, Instances filteredData) throws Exception{

		 // 속성명 설정 (header)
		 String header[] = new String[filteredData.numAttributes()];

		 for (int i = 0; i < filteredData.numAttributes() ; i++) 
			header[i] = filteredData.attribute(i).name();
		 
		 // 데이터 값 설정 (contents)
		 String contents[][] = new String[filteredData.numInstances()][filteredData.numAttributes()];

		 for(int x=0 ; x < filteredData.numInstances(); x++){
			 Instance row = filteredData.get(x);
			 for (int y = 0; y < row.numAttributes(); y++) {
				try{
					contents[x][y] = row.stringValue(y);
				}catch (IllegalArgumentException iae){
					contents[x][y] = row.value(y)+"";
				}	
			}
		}
		this.makeTable(title, header, contents);
	 }
	 
	 // swing 테이블로 결과 출력
	 public void makeTable(String title, String[] header, String[][] contents){
		 Dimension dim = new Dimension(1024,768);
		 JFrame frame = new JFrame(title);
		 frame.setLocation(100, 200);
		 frame.setPreferredSize(dim);
		 
		 JTable table = new JTable(contents, header);
		 JScrollPane scrollpane = new JScrollPane(table);
		
		 frame.add(scrollpane);
		 frame.pack();
		 frame.setVisible(true);
	 }
	 

	 // 속성선택
	public Instances attrSelClassifier(Instances data) throws Exception{
				
		// 목표변수 설정
		data.setClassIndex(data.numAttributes()-1);
		
		// 속성선택 패널객체 생성
		AttributeSelection attrSelector = new AttributeSelection();
		
		// 상관계수 속성선택 설정
		CorrelationAttributeEval attrEval = new CorrelationAttributeEval();
		Ranker search = new Ranker();
		search.setThreshold(0.2); // 임계값 설정
		attrSelector.setEvaluator(attrEval);
		attrSelector.setSearch(search);
				
		// 속성선택 실시
		attrSelector.SelectAttributes(data);
		
		// 선별된 속성만 재배열
		Instances attrSelect = attrSelector.reduceDimensionality(data);
		return attrSelect;
	}

}
