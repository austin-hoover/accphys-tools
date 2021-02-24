import sys
import os
import math

from orbit.teapot.teapot import NodeTEAPOT

class WS_Node(NodeTEAPOT):
	"""
	Generates the Histograms for distribution in x,y, and z directions.
	It is not parallel !!!!
	"""
	def __init__(self,name = "WS"):
		NodeTEAPOT.__init__(self,name)
		self.x_min = 0.
		self.x_step = 1.
		self.n_x_points = 50
		self.y_min = 0.
		self.y_step = 1.
		self.n_y_points = 50
		self.z_min = 0.
		self.z_step = 1.
		self.n_z_points = 50
		self.setHistPoints(self.n_x_points, self.n_y_points, self.n_z_points)

	def setHistPoints(self, n_x_points = 50, n_y_points = 50, n_z_points = 50):
		self.n_x_points = n_x_points
		self.n_y_points = n_y_points
		self.n_z_points = n_z_points
		self.makeArrays()
		
	def makeArrays(self):
		self.hist_x_arr = []
		self.hist_y_arr = []
		self.hist_z_arr = []
		for i in range(self.n_x_points):
			self.hist_x_arr.append(0.)
		for i in range(self.n_y_points):
			self.hist_y_arr.append(0.)
		for i in range(self.n_z_points):
			self.hist_z_arr.append(0.)
		
	def track(self, paramsDict):
		"""
		The WS_AccNode class implementation of the AccNode class track(probe) method.
		"""
		bunch = paramsDict["bunch"]
		self.makeArrays()
		if(bunch.getSize() == 0): return
		x_max =  bunch.x(0)
		x_min =  bunch.x(0)
		y_max =  bunch.y(0)
		y_min =  bunch.y(0)
		z_max =  bunch.z(0)
		z_min =  bunch.z(0)
		for i in range(bunch.getSize()):
			(x,y,z) = (bunch.x(i),bunch.y(i),bunch.z(i))
			if(x_min > x): x_min = x
			if(x_max < x): x_max = x
			if(y_min > y): y_min = y
			if(y_max < y): y_max = y
			if(z_min > z): z_min = z
			if(z_max < z): z_max = z
		self.x_min = x_min
		self.y_min = y_min
		self.z_min = z_min
		self.x_step = (x_max - x_min)/(self.n_x_points-1)
		self.y_step = (y_max - y_min)/(self.n_y_points-1)
		self.z_step = (z_max - z_min)/(self.n_z_points-1)
		#--------------------------------
		particleAttr = "macrosize"
		has_macrosize = 0
		#--------------------------------
		if(bunch.hasPartAttr(particleAttr)): has_macrosize = 1
		for i in range(bunch.getSize()):
			(x,y,z) = (bunch.x(i),bunch.y(i),bunch.z(i))
			ind_x = int((x-x_min)/self.x_step)
			ind_y = int((y-y_min)/self.y_step)
			ind_z = int((z-z_min)/self.z_step)
			macro_size = 1.0
			if(has_macrosize):
				macro_size = bunch.partAttrValue(particleAttr, i, 0)
			self.hist_x_arr[ind_x] += abs(macro_size)
			self.hist_y_arr[ind_y] += abs(macro_size)
			self.hist_z_arr[ind_z] += abs(macro_size)
		sum_hist_x = 0.
		for ix in range(len(self.hist_x_arr)):
			sum_hist_x += math.fabs(self.hist_x_arr[ix])*self.x_step*1000.
		for ix in range(len(self.hist_x_arr)):
			self.hist_x_arr[ix] /= sum_hist_x
		sum_hist_y = 0.
		for iy in range(len(self.hist_y_arr)):
			sum_hist_y += math.fabs(self.hist_y_arr[iy])*self.y_step*1000.
		for iy in range(len(self.hist_y_arr)):
			self.hist_y_arr[iy] /= sum_hist_y
		sum_hist_z = 0.
		for iz in range(len(self.hist_z_arr)):
			sum_hist_z += math.fabs(self.hist_z_arr[iz])*self.z_step*1000.
		for iz in range(len(self.hist_z_arr)):
			self.hist_z_arr[iz] /= sum_hist_z
		
	def writeToASCII(self, prefix = "", suffix = ""):
		name_x = prefix+self.getName().replace(":","_")+"_x"+suffix+".dat"
		name_y = prefix+self.getName().replace(":","_")+"_y"+suffix+".dat"
		name_z = prefix+self.getName().replace(":","_")+"_z"+suffix+".dat"
		name_arr = [name_x,name_y,name_z]
		hist_arr = [self.hist_x_arr,self.hist_y_arr,self.hist_z_arr]
		step_arr = [self.x_step,self.y_step,self.z_step]
		val_min_arr = [self.x_min,self.y_min,self.z_min]
		for ind, name in enumerate(name_arr):
			name = name_arr[ind]
			hist = hist_arr[ind]
			step = step_arr[ind]
			val_min = val_min_arr[ind]
			fl_out = open(name,"w")
			for ih in range(len(hist)):
				val = val_min + (ih+0.5)*step
				s = " %3d %12.5g %12.5g "%(ih,val*1000.,hist[ih])
				#s = " %12.5g, %12.5g "%(val*1000.,hist[ih])
				fl_out.write(s+"\n")
			fl_out.close()
			
	def getWFsXYZ(self):
		hist_arr = [self.hist_x_arr,self.hist_y_arr,self.hist_z_arr]
		step_arr = [self.x_step,self.y_step,self.z_step]
		val_min_arr = [self.x_min,self.y_min,self.z_min]
		res_wfs_arr = []
		for ind in range(3):
			wf = Function()
			hist = hist_arr[ind]
			step = step_arr[ind]
			val_min = val_min_arr[ind]
			for ih in range(len(hist)):
				x = (val_min + (ih+0.5)*step)*1000.
				y = 1.0*hist[ih]
				wf.add(x,y)
			res_wfs_arr.append(wf)
		return res_wfs_arr
		
	def getRMSs(self):
		""" returns rms for x,y,z distribution. Values are in mm """
		rms_arr = [0.,0.,0.]
		sum_arr = [0,0,0]
		hist_arr = [self.hist_x_arr,self.hist_y_arr,self.hist_z_arr]
		step_arr = [self.x_step,self.y_step,self.z_step]
		val_min_arr = [self.x_min,self.y_min,self.z_min] 		
		for ind in range(len(hist_arr)):
			hist = hist_arr[ind]
			step = step_arr[ind]
			val_min = val_min_arr[ind]
			for ih in range(len(hist)):
				val = val_min + (ih+0.5)*step
				sum_arr[ind] += hist[ih]
				rms_arr[ind] += val*hist[ih]
			if sum_arr[ind] == 0:
				sum_arr[ind] = 1
			avg_val = rms_arr[ind]/sum_arr[ind]
			rms_arr[ind] = 0.
			for ih in range(len(hist)):
				val = val_min + (ih+0.5)*step
				rms_arr[ind] += (val-avg_val)**2*hist[ih]
			rms_arr[ind] /= sum_arr[ind]
			rms_arr[ind] = math.sqrt(rms_arr[ind])
		return (rms_arr[0]*1000.,rms_arr[1]*1000.,rms_arr[2]*1000.)




