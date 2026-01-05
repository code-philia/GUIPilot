import numpy as np

try:
    import networkx as nx
except:
    pass
from fitz import Point, Rect

__all__ = ["rect", "point", "line", "line_seq"]


X = 1080
Y = 2400


class Widget:
    def __init__(self, label, bounds, idx=0, is_removed=False) -> None:
        self.bounds = rect(bounds)
        self.label = label
        self.is_removed = is_removed
        self.idx = idx
        self.norm_bounds = None

    @property
    def center(self):
        return self.bounds.center

    @property
    def norm_center(self):
        if self.norm_bounds is None:
            return self.center
        return self.norm_bounds.center

    @property
    def shape(self):
        return self.bounds.shape

    def normalize(self, width, height=None):
        self.width = width
        self.height = height
        if self.height:
            t = [width, height, width, height]
            self.norm_bounds = rect([self.bounds[i] / t[i] for i in range(4)])
        else:
            self.norm_bounds = rect([b / width for b in self.bounds])

    def get_match_label(self):
        return self.is_removed, self.label, list(self.bounds)

    def __repr__(self) -> str:
        return f"{self.label} {self.bounds}"

    @staticmethod
    def from_dict(data: dict):
        return Widget(data.get("label", None), data["bounds"], data["id"])

    @staticmethod
    def from_labelme(data: dict, idx=0):
        [[x1, y1], [x2, y2]] = data["points"]
        bounds = [x1, y1, x2, y2]
        return Widget(data["label"], bounds, idx)

    def to_label_dict(self):
        return {"label": self.label, "bounds": self.bounds.to_list(), "id": self.idx}

    @staticmethod
    def from_yolo(line, id2label):
        data = line.split("\n")[0].split(" ")
        label = id2label[int(data[0])]
        x, y, w, h = [float(x) for x in data[1:]]
        bounds = [(x - w / 2) * X, (y - h / 2) * Y, (x + w / 2) * X, (y + h / 2) * Y]
        bounds = [int(x) for x in bounds]
        return Widget(label, bounds)

    def to_yolo(self, label2id, image_width, image_height):
        return Widget._to_yolo_label(
            self.label, self.bounds, label2id, image_width, image_height
        )

    @staticmethod
    def write_yolo_labels(file, label2id, widgets, image_width, image_height):
        lables = ""
        if isinstance(widgets, list):
            for widget in widgets:
                lables = (
                    lables + widget.to_yolo(label2id, image_width, image_height) + "\n"
                )
        elif isinstance(widgets, dict):
            for label, widget in zip(widgets["cls"], widgets["box"]):
                lables = (
                    lables
                    + Widget._to_yolo_label(
                        label, widget, label2id, image_width, image_height
                    )
                    + "\n"
                )
        with open(file, "w") as f:
            f.write(lables)

    @staticmethod
    def get_datasets(file, id2label):
        labels = []
        with open(file) as f:
            data = f.readlines()
        for line in data:
            labels.append(Widget.from_yolo(line, id2label))
        return labels

    @staticmethod
    def _to_yolo_label(label, bounds, label2id, image_width, image_height):
        x = (bounds[0] + bounds[2]) / (2 * image_width)
        y = (bounds[1] + bounds[3]) / (2 * image_height)
        w = (bounds[2] - bounds[0]) / image_width
        h = (bounds[3] - bounds[1]) / image_height
        return f"{label2id[label]} {x} {y} {w} {h}"


class rect(Rect):
    def __init__(self, *args):
        if len(args) == 1:
            l = args[0]
            if hasattr(l, "__getitem__") and len(l) == 4 and isinstance(l[0], line):
                return
        super().__init__(*args)

    def morm(self, w, h):
        return rect(self.x0 / w, self.y0 / h, self.x1 / w, self.y1 / h)

    def __str__(self):
        return str(tuple([round(i, 2) for i in self]))

    def __repr__(self):
        return str(self)

    def scale(self, x_scale, y_scale):
        self.x0 = self.x0 * x_scale
        self.y0 = self.y0 * y_scale
        self.x1 = self.x1 * x_scale
        self.y1 = self.y1 * y_scale

    @property
    def shape(self):
        return self.width, self.height

    @property
    def center(self):
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2

    def to_list(self):
        return [self.x0, self.y0, self.x1, self.y1]

    @property
    def T(self):
        return rect(self.y0, self.x0, self.y1, self.x1)

    def distance_to(self, r: "rect"):
        return min([i.distance_to(self) for i in r.quad])

    def intersect_all(self, bboxes):
        if len(bboxes) == 0:
            return False
        if not isinstance(bboxes, np.ndarray):
            bboxes = np.array(bboxes)
        self_box = np.array(self)
        tl = np.maximum(bboxes[:, :2], self_box[:2])
        br = np.minimum(bboxes[:, 2:], self_box[2:])
        area_i = np.prod(br - tl, axis=1) * (tl < br).all(axis=1)
        idx = np.where(abs(area_i) > 0.1)[0]
        if len(idx) > 0:
            return True
        return False

    def norm1_dist(self, other):
        self_array = np.array([self.x0, self.y0, self.x1, self.y1])
        return np.sum(np.abs(self_array - np.array(other)))

    def calc_all_iou(self, bboxes: np.ndarray):
        if isinstance(bboxes, rect):
            bboxes = [bboxes]
        if not isinstance(bboxes, np.ndarray):
            bboxes = np.array(bboxes)
        if len(bboxes) == 0:
            return []
        self_bbox = np.array(self)
        tl = np.maximum(bboxes[:, :2], self_bbox[:2])
        br = np.minimum(bboxes[:, 2:], self_bbox[2:])
        area_i = np.prod(br - tl, axis=1) * (tl < br).all(axis=1)
        area_u = (
            np.prod(self_bbox[2:] - self_bbox[:2])
            + np.prod(bboxes[:, 2:] - bboxes[:, :2], axis=1)
            - area_i
        )
        return area_i / area_u

    def calc_intersect(self, bboxes: np.ndarray):
        if isinstance(bboxes, rect):
            bboxes = [bboxes]
        bboxes = np.array(bboxes)
        if len(bboxes) == 0:
            return []
        if self.get_area() == 0:
            return []
        self_bbox = np.array(self)
        area_self = np.prod(self_bbox[2:] - self_bbox[:2])
        tl = np.maximum(bboxes[:, :2], self_bbox[:2])
        br = np.minimum(bboxes[:, 2:], self_bbox[2:])
        area_i = np.prod(br - tl, axis=1) * (tl < br).all(axis=1)
        area_other = np.prod(bboxes[:, 2:] - bboxes[:, :2], axis=1)
        return area_i / np.minimum(area_self, area_other)

    def calc_intersect2(self, bboxes: np.ndarray):
        if isinstance(bboxes, rect):
            bboxes = [bboxes]
        bboxes = np.array(bboxes)
        if len(bboxes) == 0:
            return []
        if self.get_area() == 0:
            return []
        self_bbox = np.array(self)
        area_self = np.prod(self_bbox[2:] - self_bbox[:2])
        tl = np.maximum(bboxes[:, :2], self_bbox[:2])
        br = np.minimum(bboxes[:, 2:], self_bbox[2:])
        area_i = np.prod(br - tl, axis=1) * (tl < br).all(axis=1)
        area_other = np.prod(bboxes[:, 2:] - bboxes[:, :2], axis=1)
        return area_i / np.minimum(area_self, area_other), area_i / np.maximum(
            area_self, area_other
        )

    def get_inner(self, bboxs: np.ndarray, threshold):
        bboxes = np.array(bboxs)
        if len(bboxes) == 0:
            return []
        self_bbox = np.array(self)
        tl = np.maximum(bboxes[:, :2], self_bbox[:2])
        br = np.minimum(bboxes[:, 2:], self_bbox[2:])
        area_i = np.prod(br - tl, axis=1) * (tl < br).all(axis=1)
        tl = bboxes[:, :2]
        br = bboxes[:, 2:]
        areas = np.prod(bboxes[:, 2:] - bboxes[:, :2], axis=1) * (tl < br).all(axis=1)
        return area_i / areas > threshold

    def calc_combine_rate(self, other, larger=False, allow_overlap=False):
        over_lap = Rect(self).intersect(other).get_area()
        if larger and self.get_area() < other.get_area():
            return -1
        union_area = self.get_area() + other.get_area() - over_lap
        if allow_overlap or over_lap == 0:
            i = union_area / Rect(self).include_rect(other).get_area()
            if i == 1:
                pass
            return i
        return -1

    def mini_dist_to(self, boxes):
        dists = [self.distance_to(b) for b in boxes]
        return np.argmin(dists)

    def mini_combine_rate(
        self,
        bbox1,
        allow_overlap=False,
        max_only=False,
        with_rate=False,
        sort=False,
        larger=False,
    ):
        f = rect if not isinstance(bbox1[0], rect) else lambda x: x
        r = np.array(
            [self.calc_combine_rate(f(b), larger, allow_overlap) for b in bbox1]
        )
        if not sort:
            return r
        if max_only:
            mini_ind = np.argmax(r)
        else:
            mini_ind = np.argsort(r)[::-1]
        if with_rate:
            return mini_ind, r[mini_ind]
        return mini_ind

    def contains(self, x):
        if isinstance(x, list) and len(x[0]) == 4:
            x = np.array(x)
            vectors = np.array(self) - np.array(x)
            tl = np.logical_and(vectors[:, 0] <= 0, vectors[:, 1] <= 0)
            br = np.logical_and(vectors[:, 2] >= 0, vectors[:, 3] >= 0)
            idx = np.where(np.logical_and(tl, br))[0]
            if len(idx) == 1:
                return idx
            elif len(idx) > 1:
                to_merge = x[idx, :]
                s_i = np.lexsort((to_merge[:, 1], to_merge[:, 0]), axis=0)
                return idx[s_i]
            else:
                return []
        else:
            return super().contains(x)

    def __lt__(self, other: "rect"):
        if self.x0 < other.x0:
            return True
        elif self.x0 == other.x0:
            return self.y0 < other.y0

    def __sub__(self, p):
        return rect(super().__sub__(p))

    def inner(self, other: "rect"):
        return (
            self.x0 > other.x0
            and self.y0 > other.y0
            and self.x1 < other.x1
            and self.y1 < other.y1
        )

    def relative_loc(self, other: "rect"):
        x = (self.x0 - other[0] + self.x1 - other[2]) / 2
        y = (self.y0 - other[1] + self.y1 - other[3]) / 2
        return point(x, y)


class point(Point):
    def __init__(self, *args):
        super().__init__(*args)


class line:
    def __init__(self, *args, directed=False, ordered=True) -> None:
        if len(args) > 4:
            raise ValueError("line: bad seq len")
        if len(args) == 4:
            self.start = point(args[0], args[1])
            self.end = point(args[2], args[3])
        if len(args) == 2:
            if isinstance(args[0], point) and isinstance(args[1], point):
                self.start = args[0]
                self.end = args[1]
            else:
                self.start = point(args[0])
                self.end = point(args[1])
        if len(args) == 1:
            l = args[0]
            if hasattr(l, "__getitem__") is False:
                raise ValueError("line: bad args")
            if len(l) != 4:
                raise ValueError("line: bad seq len")
            self.start = point(l[0], l[1])
            self.end = point(l[2], l[3])
        if (
            not directed
            and ordered
            and self.start.x >= self.end.x
            and self.start.y >= self.end.y
        ):
            a = self.start
            self.start = self.end
            self.end = a
        self.directed = directed
        self.line_vec = self.end - self.start
        self.line_len = abs(self.line_vec)

    def to_json(self):
        return [list(self.start), list(self.end)]

    @property
    def points(self):
        return [self.start, self.end]

    def reverse(self):
        return line(self.end, self.start, directed=self.directed)

    def widen(self, wide):
        a = rect(self.start, self.end)
        if abs(a.x0 - a.x1) < wide:
            x = (a.x0 + a.x1) / 2
            b = (x - wide, a.y0, x + wide, a.y1)
        elif abs(a.y0 - a.y1) < wide:
            y = (a.y0 + a.y1) / 2
            b = (a.x0, y - wide, a.x1, y + wide)
        else:
            b = [0, 0, 0, 0]
        return [int(i) for i in b]

    def extend(self, length, copy=False):
        if not self.directed:
            return False
        end_point = self.end + self.line_vec.unit * length
        end_point = point(round(end_point.x), round(end_point.y))
        if copy:
            return line(self.start, end_point, directed=True)
        self.end = end_point

    def point_to_rect(self, r: rect, extend_len=0):
        l = self.extend(extend_len, True)
        if r.contains(l.end):
            return True

    def clip_line(self, r: rect):
        EPS = 1e-6
        sp = self.start
        sq = self.end
        amin = r.tl
        amax = r.br
        d = [sq[i] - sp[i] for i in range(2)]
        tmin = 0.0
        tmax = np.inf
        for i in range(2):
            if abs(d[i]) < EPS:
                if sp[i] < amin[i] or sp[i] > amax[i]:
                    return False
            else:
                ood = 1.0 / d[i]
                t1 = (amin[i] - sp[i]) * ood
                t2 = (amax[i] - sp[i]) * ood
                if t1 > t2:
                    t1, t2 = t2, t1
                if t1 > tmin:
                    tmin = t1
                if t2 < tmax:
                    tmax = t2
                if tmin > tmax:
                    return False
        return True

    def to_rect(self):
        a = rect(self.start, self.end)
        return rect(a.tl.x, a.tl.y, a.br.x, a.br.y)

    def __getitem__(self, i):
        return (self.start.x, self.start.y, self.end.x, self.end.y)[i]

    def __len__(self):
        return 4

    def __setitem__(self, key, value):
        if key > 1 or not isinstance(value, point):
            return
        if key == 0:
            self.start = value
        elif key == 1:
            self.end = value

    def __connect_to(self, other: "line", threshold):
        if self.line_len < threshold or other.line_len < threshold:
            return False
        connected = []
        if self.directed:
            if self.end.distance_to(other.start) < threshold:
                connected.append((0, 1))
            if self.start.distance_to(other.end) < threshold:
                connected.append((1, 0))
        else:
            for i in range(2):
                for j in range(2):
                    if self.points[i].distance_to(other.points[j]) < threshold:
                        connected.append((i, j))
        if len(connected) == 0:
            return False
        if len(connected) == 1:
            return connected[0]
        if len(connected) > 1:
            return 1

    def close_to(self, a_point, threshold):
        if self.start.distance_to(a_point) < threshold:
            return True
        elif self.end.distance_to(a_point) < threshold:
            return True
        return False

    def connect_to(self, other: "line", threshold, merge=False):
        p = self.__connect_to(other, threshold)
        if merge:
            if isinstance(p, tuple):
                self[p[0]] = other.points[p[1]]
        return p

    def similar_to(self, others, line_wide):
        d = np.full(4, line_wide) * 3
        return np.where((np.abs(np.array(others) - np.array(self)) < d).all(axis=1))[0]

    def len_in_range(self, line_range):
        if (
            self.line_len < line_range[0]
            or len(line_range) > 1
            and self.line_len > line_range[1]
        ):
            return False
        return True

    def distance_to(self, x):
        if isinstance(x, line):
            return self.__distance_to_line(x)
        if isinstance(x, point):
            return self.__distance_to_point(x)
        if isinstance(x, rect):
            return self.__distance_to_rect(x)
        if len(x) == 2:
            return self.__distance_to_point(point(x))
        if len(x) == 4:
            return self.__distance_to_rect(rect(x))

    def vertical_to(self, l: "line", threshold=0):
        if np.dot(self.line_vec.unit, l.line_vec.unit) <= threshold:
            return True
        return False

    def __distance_to_line(self, l: "line"):
        return min([self.__distance_to_point(q)[1] for q in l.points])

    def __distance_to_rect(self, r: rect):
        return min([self.__distance_to_point(q)[1] for q in r.quad])

    def __distance_to_point(self, p: point):
        if self.line_len == 0:
            return self.start, p.distance_to(self.start)
        point_vec = p - self.start
        point_vec_scaled = point_vec / self.line_len
        t = np.dot(self.line_vec.unit, point_vec_scaled)
        nearest = self.line_vec * t
        projected = self.start + nearest
        if t >= 0 and t <= 1:
            return projected, p.distance_to(projected)
        elif t < 0:
            return projected, p.distance_to(self.start)
        elif t > 1:
            return projected, p.distance_to(self.end)

    def __repr__(self):
        return "Line" + str(tuple(self.start)) + str(tuple(self.end))

    def __hash__(self) -> int:
        return hash((self.start.x, self.start.y, self.end.x, self.end.y))


class line_seq:
    def __init__(self, threshold) -> None:
        self.g = nx.Graph()
        self.threshold = threshold

    def add_line(self, new_l: line):
        mapping = {}
        G = nx.Graph([(new_l.start, new_l.end, {"weight": new_l.line_len})])
        for p in new_l.points:
            for n in self.g.nodes():
                if p == n:
                    continue
                elif p.distance_to(n) < self.threshold:
                    mapping[p] = n
        if new_l.line_len < self.threshold and len(mapping) > 1:
            mapping.pop(p)
        nx.relabel_nodes(G, mapping, copy=False)
        self.g = nx.compose(self.g, G)

    def add_lines(self, lines):
        for l in lines:
            self.add_line(l)

    def to_point_flow(self):
        G = nx.DiGraph()
        n_d = list(self.g.degree())
        degree = [d for _, d in n_d]
        if 1 in degree:
            target = n_d[np.argmax(degree)][0]
            sources = [n for n, d in n_d if d == 1]
            paths = [nx.astar_path(self.g, s, target) for s in sources]
            l = [nx.dijkstra_path_length(self.g, s, target) for s in sources]
            path = paths[np.argmax(l)]
            for i in range(len(path) - 1):
                G.add_edge(path[i], path[i + 1])
        else:
            pass
        return G
