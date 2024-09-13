#include <stdint.h>
#include <math.h>

#include <dlfcn.h>

#define M_TAU (2*M_PI)

#include <array>
#include <vector>
#include <map>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <pybind11/embed.h>
#include <pybind11/numpy.h>

#include <onnxruntime_cxx_api.h>

namespace py = pybind11;
using namespace py::literals;

namespace upo_laser_people_detector {

namespace {

	static const char python_code[] =
R"(
import numpy as np
from scipy.signal import find_peaks

def parse_seg(scan, scan_xy, seg, hardcoded_radius):
	peaks, props = find_peaks(seg, height=0.01, prominence=0.1, width=1, rel_height=0.5)
	if len(peaks)==0: return None, []

	left_ips = props['left_ips']
	right_ips = props['right_ips']
	peak_heights = props['peak_heights']

	scores = []
	centers = []
	counts = []

	for i,peak in enumerate(peaks):
		left_idx = int(np.ceil(left_ips[i]))
		right_idx = int(np.floor(right_ips[i]))
		peak_point = scan_xy[peak,:]
		scores.append(peak_heights[i])
		counts.append(right_idx - left_idx + 1)
		if left_idx==right_idx:
			centers.append(peak_point)
			continue

		# Find the closest point
		closest_point_idx = np.argmin(scan[left_idx:right_idx]) + left_idx
		closest_point = scan_xy[closest_point_idx,:]

		# Check if the peak is far from the closest point (it could be a point in the middle of two legs)
		pdist = peak_point - closest_point
		pdist = np.hypot(pdist[0], pdist[1])
		if pdist <= hardcoded_radius:
			centers.append(peak_point)
			continue

		# Otherwise: compute a new center close to the closest point
		all_points = scan_xy[left_idx:right_idx,:]
		pdist = all_points[:,None,:] - closest_point[None,None,:]
		pdist = np.hypot(pdist[:,0,0], pdist[:,0,1])
		chosen_points = all_points[np.nonzero(pdist <= 2*hardcoded_radius)[0],:]
		centers.append(np.mean(chosen_points, axis=0))

	scores = np.stack(scores, axis=0)
	centers = np.stack(centers, axis=0)
	return np.concatenate([ scores[:,None], centers ], axis=1), counts

def join_dets(dets, counts, radius):
	out = []
	cnt = []

	def find_close_point(pt):
		for i,ref in enumerate(out):
			d = pt[1:3] - ref[1:3]
			d = np.hypot(d[0], d[1])
			if d <= radius: return i
		return None

	# This is basically NMS, but merging instead of discarding
	for i in np.argsort(-dets[:,0], kind='stable'):
		cur_det = dets[i,:]
		cur_cnt = counts[i]

		j = find_close_point(cur_det)
		if j is not None:
			# Merge with detection
			other_det = out[j]
			other_cnt = cnt[j]
			cnt[j] += cur_cnt
			alpha = float(cur_cnt) / float(cur_cnt + other_cnt)
			other_det[:] = alpha * cur_det + (1.-alpha) * other_det

			# Ensure detection array stays sorted
			out.sort(key=lambda r: r[0], reverse=True)
		else:
			# New point, add it to the list
			out.append(cur_det)
			cnt.append(cur_cnt)

	return np.stack(out, axis=0)
)";

	visualization_msgs::msg::Marker makeEmptyMarker(std::string const& ref)
	{
		visualization_msgs::msg::Marker ret{};
		ret.header.frame_id = ref;
		ret.action = visualization_msgs::msg::Marker::DELETEALL;
		return ret;
	}

	visualization_msgs::msg::Marker makePersonMarker(std::string const& ref, int32_t id, float x, float y, float radius)
	{
		visualization_msgs::msg::Marker ret{};
		ret.header.frame_id = ref;
		ret.id = id;
		ret.type = visualization_msgs::msg::Marker::CYLINDER;
		ret.action = visualization_msgs::msg::Marker::ADD;
		ret.pose.position.x = x;
		ret.pose.position.y = y;
		ret.scale.x = 2.0f*radius;
		ret.scale.y = 2.0f*radius;
		ret.scale.z = 1.5f;
		ret.color.r = 1.0f;
		ret.color.g = 0.0f;
		ret.color.b = 0.0f;
		ret.color.a = 0.5f;
		return ret;
	}

	constexpr unsigned pad_size(unsigned size, unsigned mult)
	{
		size += mult - 1;
		size -= size % mult;
		return size;
	}

}

class LaserModelPeaks final {

	rclcpp::Logger m_log;

	Ort::Session m_session{nullptr};
	std::optional<Ort::Allocator> m_allocator{};
	std::optional<Ort::IoBinding> m_binding{};

	py::dict m_pyScope{};

	float m_scanNear, m_scanFar, m_scoreThresh, m_personRadius;

public:

	using Person = std::array<float, 3>;

	LaserModelPeaks(rclcpp::Logger&& log, Ort::Env& env, std::string const& modelPath, float near, float far, float thresh, float person_radius) :
		m_log{std::move(log)}, m_scanNear{near}, m_scanFar{far}, m_scoreThresh{thresh}, m_personRadius{person_radius}
	{
		Ort::SessionOptions options{};
		options.AppendExecutionProvider_CUDA(OrtCUDAProviderOptions{});
		m_session = Ort::Session(env, modelPath.c_str(), options);
		m_allocator.emplace(m_session, Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
		m_binding.emplace(m_session);

		m_binding->BindOutput("output", m_allocator->GetInfo());

		py::exec(python_code, m_pyScope);
	}

	std::vector<Person> infer(sensor_msgs::msg::LaserScan const& msg)
	{
		int64_t scan_shape[] = {
			1, (int64_t)pad_size(msg.ranges.size(), 6), 1
		};

		Ort::Value scan = Ort::Value::CreateTensor<float>(*m_allocator, &scan_shape[0], 3);
		float* scan_data = scan.GetTensorMutableData<float>();
		for (size_t i = 0; i < msg.ranges.size(); i ++) {
			float x = msg.ranges[i] / m_scanFar;
			if (x < 0.0f) x = 0.0f;
			else if (x > 1.0f) x = 1.0f;
			scan_data[i] = 1.0f - x;
		}
		for (size_t i = msg.ranges.size(); i < (size_t)scan_shape[1]; i ++) {
			scan_data[i] = 0.0f;
		}
		m_binding->BindInput("scan", scan);

		RCLCPP_DEBUG(m_log, "[LFE-Peaks] Inference start");
		m_binding->SynchronizeInputs();
		m_session.Run(Ort::RunOptions{nullptr}, *m_binding);
		m_binding->SynchronizeOutputs();
		RCLCPP_DEBUG(m_log, "[LFE-Peaks] Inference end");

		auto seg_out = std::move(m_binding->GetOutputValues()[0]);
		const float* seg_raw = seg_out.GetTensorData<float>();
		auto seg_shape = std::move(seg_out.GetTensorTypeAndShapeInfo().GetShape());

		if (seg_shape.size() != sizeof(scan_shape)/sizeof(scan_shape[0]) || memcmp(seg_shape.data(), scan_shape, sizeof(scan_shape)) != 0) {
			RCLCPP_ERROR(m_log, "[LFE-Peaks] Unexpected segmentation shape");
			return {};
		}

		py::array_t<float> py_scan{ (py::ssize_t)msg.ranges.size(), msg.ranges.data() };
		py::array_t<float> py_seg{ (py::ssize_t)msg.ranges.size(), seg_raw };

		py::ssize_t py_scan_xy_shape[] = { (py::ssize_t)msg.ranges.size(), 2 };
		py::array_t<float> py_scan_xy{py_scan_xy_shape};
		for (py::ssize_t i = 0; i < py_scan_xy_shape[0]; i ++) {
			float angle = msg.angle_min + i*msg.angle_increment;
			py_scan_xy.mutable_at(i, 0) = msg.ranges[i] * cosf(angle);
			py_scan_xy.mutable_at(i, 1) = msg.ranges[i] * sinf(angle);
		}

		auto [py_peaks, py_counts] = m_pyScope["parse_seg"](py_scan, py_scan_xy, py_seg, m_personRadius).cast<std::pair<py::array_t<float>, py::list>>();
		if (!py_peaks || !py_peaks.ndim()) {
			RCLCPP_DEBUG(m_log, "[LFE-Peaks] No peaks detected");
			return {};
		}

		auto py_dets = m_pyScope["join_dets"](py_peaks, py_counts, 1.5f*m_personRadius).cast<py::array_t<float>>();

		std::vector<Person> people;
		for (py::ssize_t i = 0; i < py_dets.shape(0); i ++) {
			float score = py_dets.at(i,0);
			if (score < m_scoreThresh) {
				continue;
			}

			people.emplace_back(Person{ score, py_dets.at(i,1), py_dets.at(i,2) });
		}

		return people;
	}

};

class LaserModelHostPeaks : public rclcpp::Node {

	py::scoped_interpreter m_pythonGuard{false, 0, nullptr, false};
	Ort::Env m_ortEnv{ORT_LOGGING_LEVEL_WARNING, "lasermodelhostpeaks", OrtRosLogging, this};
	std::optional<LaserModelPeaks> m_model;

	rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr m_sub{};
	std::shared_ptr<rclcpp::Publisher<visualization_msgs::msg::MarkerArray>> m_pub{};

	void* m_hPythonDylib{};

	static void OrtRosLogging(
		void* param, OrtLoggingLevel severity,
		const char* category, const char* logid, const char* code_location, const char* message)
	{
		// ORT logging severity levels map exactly to ROS logging levels
		RCUTILS_LOG_COND_NAMED(severity, RCUTILS_LOG_CONDITION_EMPTY, RCUTILS_LOG_CONDITION_EMPTY,
			static_cast<LaserModelHostPeaks*>(param)->get_logger().get_name(),
			"[ONNX %s] %s", category, message);
	}

	void onLaserScan(sensor_msgs::msg::LaserScan const& msg)
	{
		auto people = m_model->infer(msg);

		auto out = std::make_unique<visualization_msgs::msg::MarkerArray>();

		out->markers.emplace_back(makeEmptyMarker(msg.header.frame_id));

		for (size_t i = 0; i < people.size(); i ++) {
			auto& p = people[i];
			out->markers.emplace_back(makePersonMarker(msg.header.frame_id, 1+i, p[1], p[2], 0.4f));
		}

		m_pub->publish(std::move(out));
	}

public:

	LaserModelHostPeaks(rclcpp::NodeOptions const& options) :
		Node{"lasermodelhostpeaks", options}
	{
		Dl_info dlinfo;
		if (dladdr(PyExc_RecursionError, &dlinfo)) {
			RCLCPP_DEBUG(get_logger(), "Pinning %s", dlinfo.dli_fname);
			m_hPythonDylib = dlopen(dlinfo.dli_fname, RTLD_LAZY | RTLD_GLOBAL);
		}

		auto model_file = declare_parameter<std::string>("model_file");

		auto laser_topic = declare_parameter<std::string>("laser_topic", "/scanfront");
		auto marker_topic = declare_parameter<std::string>("marker_topic", "detected_people");

		auto near   = declare_parameter<float>("scan_near",       0.02f);
		auto far    = declare_parameter<float>("scan_far",        10.0f);
		auto thresh = declare_parameter<float>("score_threshold", 0.484925f);
		auto radius = declare_parameter<float>("person_radius",   0.4f);

		RCLCPP_DEBUG(get_logger(), "near=%f far=%f", near, far);

		m_model.emplace(get_logger(), m_ortEnv, model_file, near, far, thresh, radius);

		using std::placeholders::_1;
		m_sub = create_subscription<sensor_msgs::msg::LaserScan>(
			laser_topic, 10, std::bind(&LaserModelHostPeaks::onLaserScan, this, _1)
		);

		m_pub = create_publisher<visualization_msgs::msg::MarkerArray>(marker_topic, 10);
	}

	~LaserModelHostPeaks()
	{
		if (m_hPythonDylib) {
			dlclose(m_hPythonDylib);
		}
	}

};

}

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(upo_laser_people_detector::LaserModelHostPeaks)
