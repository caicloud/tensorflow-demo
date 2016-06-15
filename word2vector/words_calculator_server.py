import sys
import BaseHTTPServer
import numpy
from sklearn.metrics.pairwise import cosine_similarity
from urlparse import urlparse

v = []
reverse = {}
dic = {}

def isword(s):
    for i in range(len(s)):
        if s[i] <= 'z' and s[i] >= 'a':
            return True
    return False

class SimpleHTTPRequestHandler(BaseHTTPServer.BaseHTTPRequestHandler):
    def do_GET(self):
        global v
        global reverse
        global dic

        query = urlparse(self.path).query
        query_components = dict(qc.split("=") for qc in query.split("&"))
        a = query_components["a"]
        b = query_components["b"]
        c = query_components["c"]
        
        print "Get request with: ", a, b, c
        if not a in dic:
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write("Word " + a + " is not in the dictionary")
        if not b in dic:
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write("Word " + b + " is not in the dictionary")            
        if not c in dic:
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write("Word " + c + " is not in the dictionary")          
                        
        cur = v[dic[a]] - v[dic[b]] + v[dic[c]]
        sim = cosine_similarity([cur], v)[0]
        x = numpy.argsort(-sim)
        content = "<ol>\n"
        i = 0
        j = 0
        while j < 5:
            if reverse[x[i]] != a and reverse[x[i]] != c:
                if j == 0:
                    content +=  "  <li><h1 style=\"font-size:150%;\">" + reverse[x[i]] + "</h1></li>"
                else:
                    content += "  <li>" + reverse[x[i]] + "</li>\n"
                j += 1
            i += 1
        content += "</ol>"
        print "Response: " + content

        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(content)

fname = "glove.6B.300d.txt"
with open(fname) as fp:
    print "Loading w2v"
    i = 0
    for line in fp:
        paras = line.split(" ")
        if not isword(paras[0]): continue
        dic[paras[0]] = i
        reverse[i] = paras[0]
        a = numpy.asarray(map(float, paras[1:]))
        v.append(a / numpy.sqrt(sum(numpy.square(a))))
        i += 1
    print "w2v dic loaded."    

Handler = SimpleHTTPRequestHandler
Server = BaseHTTPServer.HTTPServer
Protocol = "HTTP/1.0"
if sys.argv[1:]:
    port = int(sys.argv[1])
else:
    port = 8000
server_address = ('127.0.0.1', port)
Handler.protocol_version = Protocol
httpd = Server(server_address, Handler)
print("Serving HTTP")
httpd.serve_forever()

'''
curl "http://localhost:8000?a=boy&b=girl&c=wife"
curl "http://localhost:8000?a=boy&b=girl&c=mother"
curl "http://localhost:8000?a=boy&b=girl&c=policewoman"

curl "http://localhost:8000?a=china&b=bejing&c=washington"
curl "http://localhost:8000?a=china&b=bejing&c=shanghai"
curl "http://localhost:8000?a=chicago&b=illinois&c=texas"

curl -X GET -H "a: mango" -H "b: mangoes" -H "c: men" "http://localhost:8000"

curl -X GET -H "a: europe" -H "b: euro" -H "c: dollar" "http://localhost:8000"
'''
