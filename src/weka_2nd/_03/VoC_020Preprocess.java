package weka_2nd._03;

import java.awt.Dimension;

import javax.swing.JFrame;
import javax.swing.JScrollPane;
import javax.swing.JTable;

import weka.core.Instance;
import weka.core.Instances;

public class VoC_020Preprocess {

	private Instances voc_sentence= null;
	private Instances voc_word= null;
	
	public VoC_020Preprocess (Instances voc_sentence, Instances voc_word){
		this.voc_sentence = voc_sentence;
		this.voc_word = voc_word;
	}
	
	public Instances[] filter() throws Exception{

		Instances[] data = new Instances[2];
		
		data[0] = this.setString2WordVector(this.voc_sentence);
		Instances tmp = this.setString2WordVector(this.voc_sentence);
		this.showRules("문장 데이터 , 속성개수 : " + tmp.numAttributes(), tmp);
		
		data[1] = this.setString2WordVector(this.voc_word);
		tmp = this.setString2WordVector(this.voc_word);
		this.showRules("단어 데이터 , 속성개수 : " + tmp.numAttributes(), tmp);
		
		return data;
	}	
	
	public Instances setString2WordVector (Instances voc) throws Exception{
		VoC_021Filter filter =  new VoC_021Filter(voc);
		Instances s2vec_voc = filter.setString2WorkVector()
				                    .getData();
		s2vec_voc.setClassIndex(0); // class = first attr
		return s2vec_voc;
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
}
