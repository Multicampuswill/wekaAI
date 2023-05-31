package com.multi.weka04.komoran;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import kr.co.shineware.nlp.komoran.constant.DEFAULT_MODEL;
import kr.co.shineware.nlp.komoran.core.Komoran;
import kr.co.shineware.nlp.komoran.model.KomoranResult;
import weka.filters.unsupervised.attribute.Add;

//https://needjarvis.tistory.com/744

public class KomoranTest6 {

	public static void main(String[] args) {
		Komoran komoran = new Komoran(DEFAULT_MODEL.FULL);
		String document = "이 드라마의 결과가 이것이라니... 15화까지는 흥미진진했는데, 갑자기 모든 것이 꿈이라니.. 너무 허무하다.";
		KomoranResult komoranResult = komoran.analyze(document); // 단어 분석 결과
		List<String> nounList = komoranResult.getNouns(); // 명사 목록만 추출
		System.out.println("전체 명사 목록만 추출");
		System.out.println(nounList); // 전체 명사 목록 추출
		System.out.println();

		//stopWords만들기 
		HashSet<String> stopWords = new HashSet<>();

		try {
			BufferedReader buffer = new BufferedReader(new InputStreamReader(new FileInputStream(
					"data/stop.txt")));
			String line = null;
			// System.out.println(buffer.readLine());
			while ((line = buffer.readLine()) != null) {
				if (line.trim().length() > 0) {
					stopWords.add(line.trim());
				}
			}
		} catch (Exception e) {
		}
		System.out.println(stopWords); //stopWords프린트 
		System.out.println(stopWords.size());
		
		//금칙어 제외처리
		List<String> nounList2 = new ArrayList<String>();
		for (String s : nounList) {
			if(stopWords.contains(s)) {
				continue;
			}
			nounList2.add(s);
		}
		//금칙어 제외된 명사목록 
		System.out.println(nounList2);
		
		
		
	}
}
