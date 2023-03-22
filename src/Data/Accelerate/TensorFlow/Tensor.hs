{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Use join" #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ExplicitNamespaces #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TypeOperators #-}
module Data.Accelerate.TensorFlow.Tensor where

import Data.Array.Accelerate.Representation.Shape
import Data.Array.Accelerate.Array.Buffer
import Data.Array.Accelerate.Array.Unique (UniqueArray(UniqueArray), withUniqueArrayPtr)
import Data.Array.Accelerate.Lifetime
import GHC.ForeignPtr
import Control.Monad.IO.Class (liftIO)
import Data.Array.Accelerate.Type (ScalarType (..), SingleType (NumSingleType))

import qualified TensorFlow.Ops                                     as TF
import qualified TensorFlow.Core                                    as TF
import qualified TensorFlow.Tensor                                  as TF
import qualified TensorFlow.Types                                   as TF
import qualified TensorFlow.Session                                 as TF
import Data.Vector (Vector)
import Data.Int
import qualified TensorFlow.GenOps.Core                             as TF hiding (shape, placeholder)
import Foreign (Ptr, castPtr, Word8)
import TensorFlow.Tensor
import Unsafe.Coerce (unsafeCoerce)
import Data.Array.Accelerate.Analysis.Match ( type (:~:)(Refl), matchScalarType )
import TensorFlow.Types



-- 2

toTFShape :: ShapeR sh -> sh -> TF.Shape
toTFShape shR sh = TF.Shape $ fromIntegral <$> shapeToList shR sh

-- build ipv value
-- hoe maak ik de buffer ervan?
-- hoe case match ik op TensorType?
fromBuffer :: forall sh t. ShapeR sh -> ScalarType t -> sh -> Buffer t -> TF.Tensor TF.Build t
fromBuffer shR t sh buffer = TF.constant (toTFShape shR sh) $ bufferToList t (size shR sh) buffer

toBuffer :: ScalarType t -> IO (Vector t) -> IO (Buffer t)
toBuffer t v = undefined

-- 1 Data.Array.Accelerate.AST.Execute
-- executeAfunSchedule :: GFunctionR t -> sched kernel () (Scheduled sched t) -> IOFun (Scheduled sched t)
