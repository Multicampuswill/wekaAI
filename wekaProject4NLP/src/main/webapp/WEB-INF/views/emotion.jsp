<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<title>Insert title here</title>
<%@ include file="../../header.jsp" %>
</head>
<body>
<h1>감정분석 결과</h1>
<hr color="red">
<h1 style="color:red; background: yellow;">${result}</h1>
<img src="resources/img/${result}.jpg">
</body>
</html>
