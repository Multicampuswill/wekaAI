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
	 
	public class ETRI_동임이의어 {
	 
	    static public void main ( String[] args ) {
	        String openApiURL = "http://aiopen.etri.re.kr:8000/WiseWWN/Word";
	        String accessKey = ETRI_key.key;    // 발급받은 API Key
	        String word = "할아버지";                // 정보를 조회할 어휘 데이터
	        Gson gson = new Gson();
	 
	        Map<String, Object> request = new HashMap<>();
	        Map<String, String> argument = new HashMap<>();
	 
	        argument.put("word", word);
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
