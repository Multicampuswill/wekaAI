package com.multi.weka04.komoran;

import java.util.List;

import kr.co.shineware.nlp.komoran.constant.DEFAULT_MODEL;
import kr.co.shineware.nlp.komoran.core.Komoran;
import kr.co.shineware.nlp.komoran.model.KomoranResult;
import kr.co.shineware.nlp.komoran.model.Token;

public class KomoranTest5 {

	public static void main(String[] args) {
		
		Komoran komoran = new Komoran(DEFAULT_MODEL.FULL);
		
		String strToAnalyze = "대한민국은 민주공화국이다.";

		KomoranResult analyzeResultList = komoran.analyze(strToAnalyze);

		System.out.println(analyzeResultList.getPlainText());

		List<Token> tokenList = analyzeResultList.getTokenList();
		
		for (Token token : tokenList) {
			System.out.format("(%2d, %2d) %s/%s\n", 
								token.getBeginIndex(), 
								token.getEndIndex(), 
								token.getMorph(),
								token.getPos());
		}

	}
}