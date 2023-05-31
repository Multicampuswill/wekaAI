package com.multi.weka04.komoran;

import java.util.List;

import kr.co.shineware.nlp.komoran.constant.DEFAULT_MODEL;
import kr.co.shineware.nlp.komoran.core.Komoran;
import kr.co.shineware.nlp.komoran.model.KomoranResult;
import kr.co.shineware.nlp.komoran.model.Token;

public class KomoranTest3 {

	public static void main(String[] args) {

		Komoran komoran = new Komoran(DEFAULT_MODEL.FULL);

		komoran.setFWDic("data/fwd.user");
		komoran.setUserDic("data/dic.user");

		KomoranResult result = komoran.analyze("내 청하는아이오아이출신입니다");
		System.out.println(result.getPlainText());
		System.out.println("----------------");
		List<Token> tokens = result.getTokenList();
		for (Token token : tokens) {
			System.out.println(token);
		}
		System.out.println("----------------");
		int nnpCount = 0;
		for (Token token : tokens) {
			if (token.getPos().equals("NNP")) {
				System.out.println(token.getMorph());
				nnpCount++;
			}
		}
		System.out.println("total NNP count >> " + nnpCount);
		System.out.println("----------------");

		List<String> nounList = result.getNouns();
		for (String s : nounList) {
			System.out.print(s + " ");
		}
	}
}