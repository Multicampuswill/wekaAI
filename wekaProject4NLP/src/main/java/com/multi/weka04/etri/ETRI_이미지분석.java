package com.multi.weka04.etri;

import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.HashMap;
import java.util.Map;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Base64;

import com.google.gson.Gson;

public class ETRI_이미지분석 {

	static public void main(String[] args) {
		String openApiURL = "http://aiopen.etri.re.kr:8000/ObjectDetect";
		String accessKey = ETRI_key.key; // 발급받은 API Key
		String type = "jpg"; // 이미지 파일 확장자
		String file = "/Users/administrator/Documents/workspace-spring-tool-suite-4-4.12.1.RELEASE/springWeka1/src/main/webapp/resources/img/etri1.jpg"; // 이미지
																																							// 파일
																																							// 경로
		String imageContents = "";
		Gson gson = new Gson();

		Map<String, Object> request = new HashMap<>();
		Map<String, String> argument = new HashMap<>();

		try {
			Path path = Paths.get(file);
			byte[] imageBytes = Files.readAllBytes(path);
			imageContents = Base64.getEncoder().encodeToString(imageBytes);
			System.out.println(imageContents);
		} catch (IOException e) {
			e.printStackTrace();
		}

		argument.put("type", type);
		argument.put("file", imageContents);

		request.put("argument", argument);

		URL url = null;
		Integer responseCode = null;
		String responBody = null;
		try {
			url = new URL(openApiURL);
			HttpURLConnection con = (HttpURLConnection) url.openConnection();
			con.setRequestMethod("POST");
			con.setDoOutput(true);
			con.setRequestProperty("Authorization", accessKey);
			DataOutputStream wr = new DataOutputStream(con.getOutputStream());
			wr.write(gson.toJson(request).getBytes("UTF-8"));
			wr.flush();
			wr.close();
			System.out.println("1. -----------------");

			responseCode = con.getResponseCode();
			System.out.println(responseCode);
			InputStream is = con.getInputStream();
			byte[] buffer = new byte[is.available()];
			int byteRead = is.read(buffer);
			responBody = new String(buffer);
			System.out.println("2. -----------------");

			System.out.println("[responseCode] " + responseCode);
			System.out.println("[responBody]");
			System.out.println(responBody);
			System.out.println("3. -----------------");

		} catch (MalformedURLException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
