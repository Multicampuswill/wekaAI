package weka_2nd._05.bird;

import weka.core.Instances;
import weka.filters.unsupervised.instance.imagefilter.*;
import weka_2nd.MoreWekaCommon;
import weka_2nd._01.Titanic_021Filter;

public class Image_021Filter extends Titanic_021Filter {	
	private AbstractImageFilter imgFilter = null;
	private String imgPath = "";

	public Image_021Filter(Instances data) {
		super(data);
		this.imgFilter = new ColorLayoutFilter();
	}
	
	public Instances action() throws Exception{
		return super.getData();
	}	
	

	public Image_021Filter setImageFilters() throws Exception {
		
		// 사진이 저장된 디렉토리
		this.imgFilter.setImageDirectory(MoreWekaCommon.imgPath+"\\bird");
		
		super.runFilter(this.imgFilter);
		
		return this;
	}
	
	public Instances getData() {
		return super.getData();
	}

	public AbstractImageFilter getImgFilter() {
		return imgFilter;
	}

	public void setImgFilter(AbstractImageFilter imgFilter) {
		this.imgFilter = imgFilter;
	}

	public String getImgPath() {
		return imgPath;
	}

	public void setImgPath(String imgPath) {
		this.imgPath = imgPath;
	}
}
