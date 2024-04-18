from flask import Flask, jsonify
app = Flask(__name__)
@app.route('/readfile', methods=['GET'])
def readfile():
    with open(r'C:\data\KG\data.txt', 'r') as f:
        content = f.read()
        return jsonify({'status': 'success', 'content': content})
if __name__ == '__main__':
    app.run(debug=True)
