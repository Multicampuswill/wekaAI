package weka_2nd._02;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.*;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka_2nd.MoreWekaCommon;

/**
 * *
 * @author bulle
 * 
 * 2 jars download : http://www.rforge.net/Rserve/files/ (REngine.jar, RserveEngine.jar)
 * 
 * 1st Rserve run	
 * 		install.packages("Rserve")
 * 		library(Rserve)
 *      Rserve()
 *
 * 2nd run Java
 * 
 * 참고 : 
 * https://niceguy1575.tistory.com/23
 * https://niceguy1575.tistory.com/25?category=738479
 * https://datamod.tistory.com/17
 */

public class Agency_021melt {
	private Instances castedData = null;
	private Instances meltedData = null;

	public Agency_021melt(Instances data) {
		this.setCastedData(data);
	}
	
	
    public Instances melt(String fixedIndex) throws Exception{
    	
    	String[] split = fixedIndex.split("-");
    	
    	// weka 구조 (1부터 시작) 와 java 데이터 구조 (0부터 시작) 의 index 차이 보정
    	int start = Integer.parseInt(split[0]) - 1; 
    	int end = Integer.parseInt(split[1]) - 1; 
    	
    	Instances castingInstance = this.getCastedData();
    	
    	// 세로형 arff 파일 생성을 위한 속성구조 생성
    	Attribute[] attr = new Attribute[castingInstance.numAttributes()];
    	ArrayList<Attribute> attrList = new ArrayList<Attribute> ();
    	    	
    	for (int x=start ; x <= end ; x++){
    		attr[x] = castingInstance.attribute(x);
    		attrList.add(attr[x]);
    	}
    	
    	ArrayList<String> variable = new ArrayList<String> ();
    	for (int y = end+1 ; y < castingInstance.numAttributes() ; y++) 
    		variable.add(castingInstance.attribute(y).name());
    	
    	// melt될 arff 데이터 구조
    	Instances meltingInstances = new Instances("speed_melt", attrList,1);
    	
    	// melt 후 새로 생성될 속성 2개 (화면명 + 화면별 속도)
    	meltingInstances.insertAttributeAt(new Attribute("variable",variable), meltingInstances.numAttributes());
    	meltingInstances.insertAttributeAt(new Attribute("value"), meltingInstances.numAttributes());

    	double[] castedValue = new double[meltingInstances.numAttributes()];
    	double[] meltedValue = null;

		for(int i=0 ; i < castingInstance.numInstances() ; i++){
			castedValue = new double[meltingInstances.numAttributes()];
    		Instance instanceCast = castingInstance.get(i);		
			int idx = 0;
			for(int y=0 ; y < castingInstance.numAttributes();y++){
				if(start <= y && y <= end){
					Attribute attrType = instanceCast.attribute(y);
					
					// 숫자형 
        			if( attrType.isNumeric() ){
        				double d1 = instanceCast.value(y);
        				castedValue[idx++] = d1;
        				
        			// String 형
					}else if (attrType.isString() ){
						String s2 = instanceCast.stringValue(y);
						castedValue[idx] = meltingInstances.attribute(idx++).addStringValue(s2);
		        		
		        	// 명목형
					}else if (attrType.isNominal() ){
						String s3 = instanceCast.stringValue(y);	        		
		        		Attribute attrTemp = attrList.get(idx);
		        		ArrayList<String> attrName = new ArrayList<String>();
		        		for(int a=0 ; a < attrTemp.numValues() ; a++) attrName.add(attrTemp.value(a));
		        		castedValue[idx++] = attrName.indexOf(s3);
					}	
				}else{
					idx = end + 1;
					String s4 = instanceCast.attribute(y).name();
					castedValue[idx++] = variable.indexOf(s4);
							
					double d4 = instanceCast.value(y);
					castedValue[idx] = d4;
					
					// 결측값 제외
					boolean isNaN = false;
					
					for(int b=0; b < castedValue.length ; b++)
						if ((castedValue[b]+"").equals("NaN"))
							isNaN = true;
					
					// 속성배열 재생성 (매주 중요, 재생성하지 않으면 기존값을 엎어침)
					meltedValue = new double[idx+1];
					for (int c=0; c < meltedValue.length ; c++) meltedValue[c] = castedValue[c];

					if(!isNaN) meltingInstances.add(new DenseInstance(1, meltedValue));
				}
			}
		}
		
		this.setMeltedData (meltingInstances);
		this.writeArff();
    	
    	return this.getMeltedData();
    }    
    
    public void writeArff() throws Exception{
 		BufferedWriter writer = new BufferedWriter(
 				                new FileWriter(MoreWekaCommon.arffPath+"agency\\speed_melt.arff"));
		writer.write(this.getMeltedData().toString());
		writer.close();    	
    }
    
    
	public Instances getCastedData() {return castedData;}
	public void setCastedData(Instances castedData) {this.castedData = castedData;}
	public Instances getMeltedData() {return meltedData;}
	public void setMeltedData(Instances meltedData) {this.meltedData = meltedData;}

}
