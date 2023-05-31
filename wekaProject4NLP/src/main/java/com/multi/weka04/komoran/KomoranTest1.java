package com.multi.weka04.komoran;

import java.util.List;

import javax.swing.JOptionPane;

import kr.co.shineware.nlp.komoran.constant.DEFAULT_MODEL;
import kr.co.shineware.nlp.komoran.core.Komoran;
import kr.co.shineware.nlp.komoran.model.KomoranResult;
import kr.co.shineware.nlp.komoran.model.Token;

public class KomoranTest1 {

	public static void main(String[] args) {
		Komoran komoran = new Komoran(DEFAULT_MODEL.FULL);
		// String document = "너무 이쁜 우리 아기들";
		// String document = JOptionPane.showInputDialog("문장을 입력하세요.@@");
		String document = "안녕! 오늘 삼성역 하늘이 참 예뻐! 너는 어디에 있니?";
		KomoranResult result = komoran.analyze(document);
		System.out.println(result);
		System.out.println(result.getPlainText());
		List<Token> list = result.getTokenList();
		System.out.println(list);

		// 각 단어의 품사와 시작위치/끝위치
		System.out.println("=============================");
		for (Token token : list) {
			System.out.println(token.getMorph() + "\t" + token.getPos() + "\t" + token.getBeginIndex() + "\t"
					+ token.getEndIndex());
		}
		System.out.println("=============================");
		// 명사 목록
		List<String> nounList = result.getNouns();
		System.out.println(nounList);

		for (Token token : list) {
			if (token.getPos().equals("NNP") || token.getPos().equals("NNG")) {
				System.out.println(token.getMorph() + " " + token.getPos() + " " + token.getBeginIndex() + " "
						+ token.getEndIndex());
			}
		}
		// 대명사는 몇개? 형용사는 몇 개?
	}
}
