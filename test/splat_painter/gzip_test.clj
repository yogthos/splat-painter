(ns splat-painter.gzip-test
  (:require [clojure.java.io :as io]
            [clojure.test :refer [deftest is testing]]
            [splat-painter.gzip :as gzip]))

(defn- bytes-at [path ^long off ^long n]
  (let [ba (byte-array n)]
    (with-open [i (io/input-stream (io/file path))]
      (.skip i off)
      (.read i ba))
    (mapv #(bit-and (long %) 255) ba)))

(defn- le32 [bs] (reduce + (map-indexed (fn [i b] (bit-shift-left (long b) (* 8 i))) bs)))

(def ^:private doc
  ;; the shape of the real output: one long run of near-identical elements, which is
  ;; where DEFLATE earns its keep
  (apply str "<svg>" (for [i (range 4000)]
                       (str "<circle cx=\"" i "\" cy=\"7\" r=\"2\" fill=\"url(#g3)\"/>"))))

(deftest writes-a-real-gzip-container
  (let [path (str (System/getProperty "java.io.tmpdir") "/splat-gzip-test.svgz")
        n    (gzip/spit-gz! path doc)
        size (.length (io/file path))]
    (try
      (is (= (alength (.getBytes ^String doc "UTF-8")) n) "returns the uncompressed length")
      (testing "RFC 1952 header: magic 1f 8b, method 8 (deflate), no flags"
        (is (= [0x1f 0x8b 0x08 0x00] (bytes-at path 0 4))))
      (testing "no mtime, so the same document gzips to the same bytes every save"
        (is (= [0 0 0 0] (bytes-at path 4 4))))
      (testing "the ISIZE trailer is the uncompressed length mod 2^32"
        (is (= (mod n 4294967296) (le32 (bytes-at path (- size 4) 4)))))
      (testing "and it is actually smaller"
        (is (< size (quot n 4)) "an SVG of repeated elements should compress hard"))
      (finally (io/delete-file (io/file path) true)))))

(deftest an-empty-document-still-writes-a-valid-container
  (let [path (str (System/getProperty "java.io.tmpdir") "/splat-gzip-empty.svgz")]
    (try
      (is (zero? (gzip/spit-gz! path "")))
      (is (= [0x1f 0x8b] (bytes-at path 0 2)))
      (is (= 0 (le32 (bytes-at path (- (.length (io/file path)) 4) 4))))
      (finally (io/delete-file (io/file path) true)))))

(deftest a-path-that-cannot-be-opened-throws
  (is (thrown? Exception (gzip/spit-gz! "/no/such/dir/x.svgz" "<svg/>"))))
