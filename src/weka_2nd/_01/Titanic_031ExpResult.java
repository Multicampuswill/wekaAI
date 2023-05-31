package weka_2nd._01;

import weka.classifiers.*;

public class Titanic_031ExpResult implements Comparable<Titanic_031ExpResult> {

	private Classifier model = null;
	private double pctCorrect = 0.0;
	private double[] pctCorrects = null;;
	
	public Classifier getModel() {
		return model;
	}
	public void setModel(Classifier model) {
		this.model = model;
	}
	public double getPctCorrect() {
		return pctCorrect;
	}
	public void setPctCorrect(double pctCorrect) {
		this.pctCorrect = pctCorrect;
	}
	public double[] getPctCorrects() {
		return pctCorrects;
	}
	public void setPctCorrects(double[] pctCorrects) {
		this.pctCorrects = pctCorrects;
	}

	@Override
	public int compareTo(Titanic_031ExpResult o) {
		if (pctCorrect > o.getPctCorrect())
			return 1;
		else if (pctCorrect < o.getPctCorrect())
			return -1;
			
		return 0;
	}
	
}
