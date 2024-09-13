#include <stdint.h>
#include <math.h>

#define M_TAU (2*M_PI)

#include <array>
#include <vector>
#include <map>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <onnxruntime_cxx_api.h>

namespace upo_laser_people_detector {

namespace {

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

}

class LaserModel final {

	rclcpp::Logger m_log;

	Ort::Session m_session{nullptr};
	std::optional<Ort::Allocator> m_allocator{};
	std::optional<Ort::IoBinding> m_binding{};

	float m_scanNear, m_scanFar, m_scoreThresh;

public:

	using Person = std::array<float, 3>;

	LaserModel(rclcpp::Logger log, Ort::Env& env, std::string const& modelPath, float near, float far, float thresh) :
		m_log{log}, m_scanNear{near}, m_scanFar{far}, m_scoreThresh{thresh}
	{
		Ort::SessionOptions options{};
		options.AppendExecutionProvider_CUDA(OrtCUDAProviderOptions{});
		m_session = Ort::Session(env, modelPath.c_str(), options);
		m_allocator.emplace(m_session, Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
		m_binding.emplace(m_session);

		m_binding->BindOutput("grid", m_allocator->GetInfo());
	}

	std::vector<Person> infer(sensor_msgs::msg::LaserScan const& msg)
	{
		int64_t scan_shape[] = {
			1, (int64_t)msg.ranges.size(), 1
		};

		Ort::Value scan = Ort::Value::CreateTensor<float>(*m_allocator, &scan_shape[0], 3);
		float* scan_data = scan.GetTensorMutableData<float>();
		for (size_t i = 0; i < msg.ranges.size(); i ++) {
			float x = msg.ranges[i] / m_scanFar;
			if (x < 0.0f) x = 0.0f;
			else if (x > 1.0f) x = 1.0f;
			scan_data[i] = 1.0f - x;
		}
		m_binding->BindInput("scan", scan);

		RCLCPP_DEBUG(m_log, "[LFE-PPN] Inference start");
		m_binding->SynchronizeInputs();
		m_session.Run(Ort::RunOptions{nullptr}, *m_binding);
		m_binding->SynchronizeOutputs();
		RCLCPP_DEBUG(m_log, "[LFE-PPN] Inference end");

		auto grid_out = std::move(m_binding->GetOutputValues()[0]);
		const float* grid_raw = grid_out.GetTensorData<float>();
		auto grid_shape = std::move(grid_out.GetTensorTypeAndShapeInfo().GetShape());
		if (grid_shape.size() != 4 || grid_shape[0] != 1 || grid_shape[3] != 3) {
			RCLCPP_ERROR(m_log, "[LFE-PPN] Unexpected grid shape");
			return {};
		}

		unsigned num_sectors = (unsigned)grid_shape[1];
		unsigned num_anchors_per_sector = (unsigned)grid_shape[2];
		unsigned num_anchors = num_sectors * num_anchors_per_sector;

		float anchor_depth = (m_scanFar - m_scanNear) / num_anchors_per_sector;
		float sector_ampl = (msg.angle_max - msg.angle_min) / num_sectors;

		RCLCPP_DEBUG(m_log, "[LFE-PPN] Received %u x %u grid (%.4fº,%.4f)",
			num_sectors, num_anchors_per_sector, sector_ampl*360.0f/(float)M_TAU, anchor_depth);

		std::multimap<float, unsigned> sorted;

		for (unsigned i = 0; i < num_anchors; i ++) {
			float i_score = grid_raw[i*3];
			if (i_score >= m_scoreThresh) {
				sorted.emplace(-i_score, i);
			}
		}

		std::vector<Person> people;

		for (auto& it : sorted) {
			// Decode position
			unsigned it_anchor = it.second % num_anchors_per_sector;
			unsigned it_sector = it.second / num_anchors_per_sector;

			float a_dist  = m_scanNear + (0.5f + it_anchor)*anchor_depth;
			float a_angle = msg.angle_min + (0.5f + it_sector)*sector_ampl;

			float reg_dist  = grid_raw[it.second*3+1] * anchor_depth;
			float reg_angle = grid_raw[it.second*3+2] * anchor_depth / a_dist;

			float it_dist = a_dist+reg_dist;
			float it_angle = a_angle+reg_angle;

			float it_x = it_dist * cosf(it_angle);
			float it_y = it_dist * sinf(it_angle);

			// NMS
			bool should_do = true;
			for (auto& other : people) {
				float dx = it_x - other[1];
				float dy = it_y - other[2];
				float diff = sqrtf(dx*dx + dy*dy);
				if (diff < 2*0.4f) {
					should_do = false;
					break;
				}
			}

			if (!should_do) {
				continue;
			}

			people.emplace_back(Person{ -it.first, it_x, it_y });

			RCLCPP_DEBUG(m_log, "[LFE-PPN]  (%.3f,%.3f) score=%.6f d=%.4f a=%.4fº",
				it_x, it_y, -it.first, it_dist, it_angle*360.0f/(float)M_TAU);
		}

		return people;
	}

};

class LaserModelHost : public rclcpp::Node {

	Ort::Env m_ortEnv{ORT_LOGGING_LEVEL_WARNING, "lasermodelhost", OrtRosLogging, this};
	std::optional<LaserModel> m_model;

	rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr m_sub{};
	std::shared_ptr<rclcpp::Publisher<visualization_msgs::msg::MarkerArray>> m_pub{};

	static void OrtRosLogging(
		void* param, OrtLoggingLevel severity,
		const char* category, const char* logid, const char* code_location, const char* message)
	{
		// ORT logging severity levels map exactly to ROS logging levels
		RCUTILS_LOG_COND_NAMED(severity, RCUTILS_LOG_CONDITION_EMPTY, RCUTILS_LOG_CONDITION_EMPTY,
			static_cast<LaserModelHost*>(param)->get_logger().get_name(),
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

	LaserModelHost(rclcpp::NodeOptions const& options) :
		Node{"lasermodelhost", options}
	{
		auto model_file = declare_parameter<std::string>("model_file");

		auto laser_topic = declare_parameter<std::string>("laser_topic", "/scanfront");
		auto marker_topic = declare_parameter<std::string>("marker_topic", "detected_people");

		auto near   = declare_parameter<float>("scan_near",       0.02f);
		auto far    = declare_parameter<float>("scan_far",        10.0f);
		auto thresh = declare_parameter<float>("score_threshold", 0.937660f);

		RCLCPP_DEBUG(get_logger(), "near=%f far=%f", near, far);

		m_model.emplace(get_logger(), m_ortEnv, model_file, near, far, thresh);

		using std::placeholders::_1;
		m_sub = create_subscription<sensor_msgs::msg::LaserScan>(
			laser_topic, 10, std::bind(&LaserModelHost::onLaserScan, this, _1)
		);

		m_pub = create_publisher<visualization_msgs::msg::MarkerArray>(marker_topic, 10);
	}

};

}

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(upo_laser_people_detector::LaserModelHost)
