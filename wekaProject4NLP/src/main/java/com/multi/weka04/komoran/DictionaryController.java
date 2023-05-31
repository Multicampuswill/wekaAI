package com.multi.weka04.komoran;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.RequestMapping;

@Controller
public class DictionaryController {

	@Autowired
	DictionaryService dictionaryService;

	@RequestMapping("emotion")
	public void one(String sentence, Model model) {
		System.out.println("emotion컨트롤러 >> " + sentence);
		
		String result = dictionaryService.emotion(sentence);
		model.addAttribute("result", result);
	}
}
