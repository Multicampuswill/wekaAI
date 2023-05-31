package com.multi.weka04.etri;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.HashMap;
import java.util.Map;

import com.google.gson.Gson;

public class ETRI_형태소분석 {
	 
	 
	    public static void main ( String[] args ) {
	        String openApiURL = "http://aiopen.etri.re.kr:8000/WiseQAnal";
	        String accessKey = ETRI_key.key;    // 발급받은 API Key
	        String text = "오늘 날씨 어때?";            // 분석할 질문의 텍스트 데이터
	        Gson gson = new Gson();
	 
	        Map<String, Object> request = new HashMap<>();
	        Map<String, String> argument = new HashMap<>();
	        argument.put("text", text);
	        request.put("argument", argument);
	        URL url;
	        Integer responseCode = null;
	        String responBody = null;
	        try {
	                url = new URL(openApiURL);
	                HttpURLConnection con = (HttpURLConnection)url.openConnection();
	                con.setRequestMethod("POST");
	                con.setDoOutput(true);
	                con.setRequestProperty("Content-Type", "application/json; charset=UTF-8");
	                con.setRequestProperty("Authorization", accessKey);

	 
	                DataOutputStream wr = new DataOutputStream(con.getOutputStream());
	                wr.write(gson.toJson(request).getBytes("UTF-8"));
	                wr.flush();
	                wr.close();
	 
	                responseCode = con.getResponseCode();
	                InputStream is = con.getInputStream();
	                byte[] buffer = new byte[is.available()];
	                int byteRead = is.read(buffer);
	                responBody = new String(buffer);
	 
	                System.out.println("[responseCode] " + responseCode);
	                System.out.println("[responBody]");
	                System.out.println(responBody);
	 
	        } catch (MalformedURLException e) {
	                e.printStackTrace();
	        } catch (IOException e) {
	                e.printStackTrace();
	        }
	    }
	}
